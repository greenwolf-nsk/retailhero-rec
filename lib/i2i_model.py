import json
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path
from typing import List
from datetime import datetime

import numpy as np
import implicit
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, vstack

from lib.config import ImplicitConfig
from lib.metrics import normalized_average_precision


test_start = datetime(2019, 3, 2, 0, 0, 0)


class ProductIdMap:

    def __init__(self, product_ids: list):
        self.id2product = dict(enumerate(product_ids))
        self.product2id = {v: k for k, v in self.id2product.items()}

    def to_id(self, product: str):
        return self.product2id[product]

    def to_product(self, id_: int):
        return self.id2product[id_]

    def __len__(self):
        return len(self.product2id)


def create_sparse_row(num_products: tuple, indices: list):
    return coo_matrix(
        (np.ones_like(indices, dtype=np.int8), ([0] * len(indices), indices)),
        shape=(1, num_products),
    )


def create_sparse_row_from_counter(num_products: tuple, counter: Counter):
    ids, counts = list(zip(*counter.items()))
    return coo_matrix(
        (np.array(counts, dtype=np.float64), ([0] * len(ids), ids)),
        shape=(1, num_products),
    )


def create_sparse_row_from_record(user_record: dict, product_id_map: ProductIdMap):
    product_counts = defaultdict(int)
    for transaction in user_record['transaction_history']:
        age = max(0, (test_start - datetime.fromisoformat(transaction['datetime'])).days)
        for product in transaction['products']:
            pid = product_id_map.to_id(product['product_id'])
            score = (age + 1) ** (-1 / 5)
            product_counts[pid] += score

    return create_sparse_row_from_counter(len(product_id_map), product_counts)


def create_sparse_purchases_matrix(purchases: List[dict], product_id_map: ProductIdMap) -> coo_matrix:
    rows = []
    for record in purchases:
        product_counts = Counter([
            product_id_map.to_id(product['product_id'])
            for transaction in record['transaction_history']
            for product in transaction['products']
        ])
        if product_counts:
            rows.append(create_sparse_row_from_counter(len(product_id_map), product_counts))

    return vstack(rows)


def create_i2i_sparse_matrix(
        purchases: List[dict],
        product_id_map: ProductIdMap
) -> csr_matrix:
    mat = lil_matrix((len(product_id_map), len(product_id_map)), dtype=np.int8)

    for record in purchases:
        for transaction in record['transaction_history']:
            product_ids = [
                product['product_id']
                for product in transaction['products']
            ]
            for product_a, product_b in combinations(product_ids, 2):
                idx_a, idx_b = product_id_map.to_id(product_a), product_id_map.to_id(product_b)
                mat[idx_a, idx_b] += 1
                mat[idx_b, idx_a] += 1

    return mat.tocsr()


def load_item_vectors(fp: Path):
    with open(fp, 'r') as f:
        vectors = json.load(f)

    return {k: np.array(v) for k, v in vectors.items()}


class ImplicitRecommender:

    def __init__(
        self,
        model,
        product_id_map: ProductIdMap,
    ):
        self.model = model
        self.product_id_map = product_id_map

    def recommend(self, user_record: dict, filter_seen: bool = False, num_recs: int = 30) -> list:
        row = create_sparse_row_from_record(user_record, self.product_id_map)
        recs = self.model.recommend(
            userid=0,
            user_items=row.tocsr(),
            N=num_recs,
            filter_already_liked_items=filter_seen,
            recalculate_user=True
        )
        return [(self.product_id_map.to_product(rec), score) for rec, score in recs]


def train_implicit_vectors(
        train_records: list,
        config: ImplicitConfig,
        product_id_map: ProductIdMap
):
    matrix = create_sparse_purchases_matrix(train_records, product_id_map)
    model = implicit.als.AlternatingLeastSquares(
        factors=config.num_factors,
        iterations=config.epochs
    )
    model.fit(matrix.T)
    item_vectors = {
        product_id_map.to_product(i): list(map(float, factor))
        for i, factor in enumerate(model.item_factors)  # user factors, cuz in implicit its inverted
    }
    with open(config.vectors_file, 'w') as f:
        json.dump(item_vectors, f)

    return item_vectors


def validate(recommender, train_records: list, test_records: list, filter_seen: bool = False):
    gt_len = []
    scores = []
    for train_record, test_record in zip(train_records, test_records):
        try:
            recs = [x[0] for x in recommender.recommend(train_record, filter_seen)]
        except:
            recs = []
        if test_record['transaction_history']:
            next_transaction = test_record['transaction_history'][0]
            gt_items = [p['product_id'] for p in next_transaction['products']]
            scores.append(normalized_average_precision(gt_items, recs))
            gt_len.append(len(gt_items))

    return scores, gt_len
