import json
import pickle
from collections import namedtuple

import attr
from pathlib import Path
from typing import Dict


Query = namedtuple("Query", "history answers")


def parse_check_queries_file(filename: Path):
    with open(filename, "r") as f:
        lines = f.readlines()

    queries = []
    for line in lines:
        history, answer = line.split("\t")
        queries.append(Query(json.loads(history), json.loads(answer)))

    return queries


def read_products_file(filename: Path) -> dict:
    records = {}
    with open(filename, "r") as f:
        header = f.readline().strip().split(',')

        for line in f:
            record = ProductsRow(*line.strip().split(','))
            records[record.product_id] = record

    return records


def deduplicate(items: list):
    seen = set()
    dedup = []
    for item in items:
        if item not in seen:
            dedup.append(item)
            seen.add(item)

    return dedup


def maybe_float(x):
    return float(x) if x else 0


@attr.s(slots=True)
class ProductsRow:
    product_id = attr.ib()
    level_1 = attr.ib()
    level_2 = attr.ib()
    level_3 = attr.ib()
    level_4 = attr.ib()
    segment_id = attr.ib(converter=maybe_float)
    brand_id = attr.ib()
    vendor_id = attr.ib()
    netto = attr.ib(converter=maybe_float)
    is_own_trademark = attr.ib(converter=bool)
    is_alcohol = attr.ib(converter=bool)
    max_dt = attr.ib(converter=maybe_float)
    min_dt = attr.ib(converter=maybe_float)
    avg_dt = attr.ib(converter=maybe_float)
    max_q = attr.ib(converter=maybe_float)
    min_q = attr.ib(converter=maybe_float)
    avg_q = attr.ib(converter=maybe_float)
    unique_clients = attr.ib(converter=maybe_float)
    max_p = attr.ib(converter=maybe_float)
    min_p = attr.ib(converter=maybe_float)
    avg_p = attr.ib(converter=maybe_float)


def inplace_hash_join(left_list: dict, products_data: Dict[str, ProductsRow]):
    for product_id in left_list['product_id']:
        for k, v in attr.asdict(products_data[product_id]).items():
            if k != 'product_id':
                left_list[k].append(v)


def pickle_dump(fp: Path, obj: object):
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fp: Path):
    with open(fp, 'rb') as f:
        return pickle.load(f)
