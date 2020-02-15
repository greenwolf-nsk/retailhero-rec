from collections import defaultdict
from itertools import chain, groupby
from operator import itemgetter
from datetime import datetime


import pandas as pd
import numpy as np

test_start = datetime(2019, 3, 1, 0, 0, 0)


def create_features_from_transactions(users_data: list, product_vectors: dict) -> pd.DataFrame:

    features = {
        'total_pucrhases': [],
        'average_psum': [],
        'client_id': [],
        'product_id': [],
        'count': [],
        'p_tr_share': [],
        'last_transaction': [],
        'last_transaction_age': [],
        'last_product_transaction_age': [],
        'client_product_dot': []
    }

    for user_data in users_data:
        trs = user_data['transaction_history']
        if not trs:
            continue
        client_id = user_data['client_id']
        products = list(chain(*(x['products'] for x in trs)))
        product_ids = [product['product_id'] for product in products]
        client_vector = create_client_vector(product_ids, product_vectors)
        transaction_ids = list(
            chain(*([i + 1 for _ in range(len(x['products']))] for i, x in enumerate(trs)))
        )
        transaction_ages = [
            (test_start - datetime.fromisoformat(x['datetime'])).days
            for x in trs
            for _ in range(len(x['products']))
        ]
        for i, tid in enumerate(transaction_ids):
            products[i]['tid'] = tid
            products[i]['tr_age'] = transaction_ages[i]

        total_transactions = len(trs)
        average_psum = sum([tr['purchase_sum'] for tr in trs]) / total_transactions

        key = itemgetter('product_id')
        for product, part in groupby(sorted(products, key=key), key=key):
            part = list(part)
            product_count = sum([product['quantity'] for product in part])
            features['total_pucrhases'].append(total_transactions)
            features['average_psum'].append(average_psum)
            features['client_id'].append(client_id)
            features['product_id'].append(product)
            features['count'].append(product_count)
            features['p_tr_share'].append(len(part) / max(transaction_ids))
            features['last_transaction'].append(max([p['tid'] for p in part]) / max(transaction_ids))
            features['last_transaction_age'].append(transaction_ages[-1])
            features['last_product_transaction_age'].append(min([p['tr_age'] for p in part]))
            client_product_dot = 0
            if product in product_vectors:
                client_product_dot = np.dot(
                    client_vector,
                    product_vectors[product]
                )
            features['client_product_dot'].append(client_product_dot)

    return pd.DataFrame(features)


def create_target_from_transactions(test_users_transactions: list) -> pd.DataFrame:
    # just get items users bought in their first transaction of test period
    columns = {
        'client_id': [],
        'product_id': [],
    }
    for user_transactions in test_users_transactions:
        if not user_transactions['transaction_history']:
            continue
        first_transaction = user_transactions['transaction_history'][0]
        client_id = user_transactions['client_id']

        for product in first_transaction['products']:
            columns['client_id'].append(client_id)
            columns['product_id'].append(product['product_id'])

    df = pd.DataFrame(columns)
    df['target'] = 1
    return df


def create_client_vector(product_ids: list, product_vectors: dict) -> np.array:
    client_vector = np.zeros(64)
    cnt = 0
    for product_id in product_ids:
        if product_id in product_vectors:
            cnt += 1
            client_vector += product_vectors[product_id]

    return client_vector / cnt


def create_product_features_from_users_data(users_data: list) -> pd.DataFrame:
    product_stats = defaultdict(lambda: defaultdict(list))
    for record in users_data:
        client_id = record['client_id']
        for tr in record['transaction_history']:
            relative_dt = (test_start - datetime.fromisoformat(tr['datetime'])).days
            for product in tr['products']:
                product_id = product['product_id']
                quantity = product['quantity']
                product_stats[product_id]['dts'].append(relative_dt)
                product_stats[product_id]['q'].append(quantity)
                product_stats[product_id]['client_id'].append(client_id)

    product_features = defaultdict(list)

    for product_id, features in product_stats.items():
        product_features['product_id'].append(product_id)
        product_features['max_dt'].append(max(features['dts']))
        product_features['min_dt'].append(min(features['dts']))
        product_features['avg_dt'].append(sum(features['dts']) / len(features['dts']))
        product_features['max_q'].append(max(features['q']))
        product_features['min_q'].append(min(features['q']))
        product_features['avg_q'].append(sum(features['q']) / len(features['q']))
        product_features['unique_clients'].append(len(set(features['client_id'])))

    return pd.DataFrame(product_features)
