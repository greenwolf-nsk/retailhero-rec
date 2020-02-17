import json
from itertools import combinations
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, vstack


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
    product_counts = Counter([
            product_id_map.to_id(product['product_id'])
            for transaction in user_record['transaction_history']
            for product in transaction['products']
        ])
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

    def recommend(self, user_record: dict, num_recs: int = 30) -> list:
        row = create_sparse_row_from_record(user_record, self.product_id_map)
        recs = self.model.recommend(
            userid=0,
            user_items=row.tocsr(),
            N=num_recs,
            filter_already_liked_items=False,
            recalculate_user=True
        )
        return [(self.product_id_map.to_product(rec), score) for rec, score in recs]
