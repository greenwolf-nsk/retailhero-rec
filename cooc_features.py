import json
from collections import defaultdict
from itertools import combinations

import attr

from pathlib import Path

import tqdm as tqdm


@attr.s
class Record:
    client_id = attr.ib()
    transaction_id = attr.ib()
    transaction_datetime = attr.ib()
    regular_points_received = attr.ib(converter=float)
    express_points_received = attr.ib(converter=float)
    regular_points_spent = attr.ib(converter=float)
    express_points_spent = attr.ib(converter=float)
    purchase_sum = attr.ib(converter=float)
    store_id = attr.ib()
    product_id = attr.ib()
    product_quantity = attr.ib(converter=float)
    trn_sum_from_iss = attr.ib(converter=float)
    trn_sum_from_red = attr.ib()


def parse_purchases(filepath: Path):
    oc_dict = defaultdict(int)
    cooc_dict = defaultdict(lambda: defaultdict(int))
    trans = 'deafbeef'
    trans_products = []
    with open(filepath, 'r') as f:
        f.readline()
        for line in tqdm.tqdm(f):
            record = Record(*line.strip().split(','))
            if record.transaction_id != trans:
                trans = record.transaction_id
                update_cooc(trans_products, cooc_dict)
                trans_products = []

            trans_products.append(record.product_id)

    with open('cooc_dict.json', 'w') as f:
        json.dump(cooc_dict, f)


def update_cooc(trans_products: list, cooc_dict: dict):
    for product_a, product_b in combinations(trans_products, 2):
        cooc_dict[product_a][product_b] += 1


if __name__ == '__main__':
    parse_purchases('data/purchases.csv')



