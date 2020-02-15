import json
from itertools import combinations
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, vstack


def extract_product_ids(purchases: List[dict]) -> list:
    unique_product_ids = set()
    for record in purchases:
        for transaction in record['transaction_history']:
            unique_product_ids.update({
                product['product_id']
                for product in transaction['products']
            })

    return sorted(list(unique_product_ids))


def encode_product_ids(product_ids: list) -> Tuple[dict, dict]:
    int_to_product_id = dict(enumerate(product_ids))
    product_id_to_int = {v: k for k, v in int_to_product_id.items()}

    return product_id_to_int, int_to_product_id


def create_sparse_row(num_products: tuple, indices: list):
    return coo_matrix(
        (np.ones_like(indices, dtype=np.int8), ([0] * len(indices), indices)),
        shape=(1, num_products),
    )


def create_sparse_row_from_counter(num_products: tuple, counter: Counter):
    ids, counts = list(zip(*counter.items()))
    return coo_matrix(
        (np.array(counts, dtype=np.int8), ([0] * len(ids), ids)),
        shape=(1, num_products),
    )


def create_sparse_purchases_matrix(purchases: List[dict]) -> coo_matrix:
    product_ids = extract_product_ids(purchases)
    num_products = len(product_ids)
    product_id_map, _ = encode_product_ids(product_ids)
    rows = []
    for record in purchases:
        product_counts = Counter([
            product_id_map[product['product_id']]
            for transaction in record['transaction_history']
            for product in transaction['products']
        ])
        if product_counts:
            rows.append(create_sparse_row_from_counter(num_products, product_counts))

    return vstack(rows)


def create_sparse_matrix_from_purchases(purchases: List[dict]) -> csr_matrix:
    product_ids = extract_product_ids(purchases)
    num_products = len(product_ids)
    product_id_map, _ = encode_product_ids(product_ids)
    mat = lil_matrix((num_products, num_products), dtype=np.int8)

    for record in purchases:
        for transaction in record['transaction_history']:
            product_ids = [
                product['product_id']
                for product in transaction['products']
            ]
            for product_a, product_b in combinations(product_ids, 2):
                idx_a, idx_b = product_id_map[product_a], product_id_map[product_b]
                mat[idx_a, idx_b] += 1
                mat[idx_b, idx_a] += 1

    return mat.tocsr()


def load_item_vectors(fp: Path):
    with open(fp, 'r') as f:
        vectors = json.load(f)

    return {k: np.array(v) for k, v in vectors.items()}




