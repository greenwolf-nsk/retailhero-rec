import json
import sys
import time

import requests

from lib.utils import deduplicate


class Timer:

    def __init__(self, timings: list):
        self.timings = timings
        self.start = 0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timings.append(int((time.time() - self.start) * 1000))


def average_precision(actual, recommended, k=30):
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / min(len(actual), k)


def normalized_average_precision(actual, recommended, k=30):
    actual = set(actual)
    if len(actual) == 0:
        return 0.0

    ap = average_precision(actual, recommended, k=k)
    return ap


def run_queries(url, queryset_file, offset, limit):
    ap_values = []
    timings = []
    with open(queryset_file) as f:
        for line_no, line in enumerate(f):
            if line_no < offset:
                continue
            elif line_no < offset + limit:
                query_data, next_transaction = line.strip().split("\t")
                query_data = json.loads(query_data)
                test_data = json.loads(next_transaction)
                if not test_data['transaction_history']:
                    continue
                next_transaction = test_data['transaction_history'][0]
                product_ids = [p['product_id'] for p in next_transaction['products']]

                with Timer(timings):
                    resp = requests.post(url, json=query_data, timeout=5)

                if resp.status_code == 200:
                #resp.raise_for_status()
                    resp_data = resp.json()

                    # assert len(resp_data["recommended_products"]) <= 30

                    ap = normalized_average_precision(
                        product_ids, deduplicate(resp_data["recommended_products"])[:30], 30
                    )
                    ap_values.append(ap)
                else:
                    ap_values.append(0)

    print('max time: ', max(timings))
    print('avg time: ', round(sum(timings) / len(timings), 3))
    map_score = sum(ap_values) / len(ap_values)
    return map_score


if __name__ == "__main__":
    offset, limit = int(sys.argv[1]), int(sys.argv[2])
    url = 'http://localhost:8000/recommend'
    queryset_file = 'data/clients_purchases.tsv'
    score = run_queries(url, queryset_file, offset, limit)
    print("Score:", score)
