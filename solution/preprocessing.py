from itertools import chain, groupby
from operator import itemgetter
from datetime import datetime


import pandas as pd


test_start = datetime(2019, 3, 1, 0, 0, 0)


def create_features_from_transactions(users_data: list) -> pd.DataFrame:

    features = {
        'total_pucrhases': [],
        'average_psum': [],
        'client_id': [],
        'product_id': [],
        'count': [],
        'count_n': [],
        'last_transaction': [],
        'last_transaction_age': [],
        'last_product_transaction_age': [],
    }

    for user_data in users_data:
        trs = user_data['transaction_history']
        if not trs:
            continue
        client_id = user_data['client_id']
        products = list(chain(*(x['products'] for x in trs)))
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
            features['total_pucrhases'].append(total_transactions)
            features['average_psum'].append(average_psum)
            features['client_id'].append(client_id)
            features['product_id'].append(product)
            features['count'].append(len(part))
            features['count_n'].append(len(part) / max(transaction_ids))
            features['last_transaction'].append(max([p['tid'] for p in part]) / max(transaction_ids))
            features['last_transaction_age'].append(transaction_ages[-1])
            features['last_product_transaction_age'].append(min([p['tr_age'] for p in part]))

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
