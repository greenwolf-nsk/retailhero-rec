from itertools import chain, groupby
from operator import itemgetter

import numpy as np
import pandas as pd


def create_features_from_purchases(purchases):
    features = {
        'total_pucrhases': [],
        'average_psum': [],
        'client_id': [],
        'product_id': [],
        'count': [],
        'count_n': [],
        'last_transaction': [],
    }
    target = []
    client_bounds = np.where(purchases.client_id != purchases.client_id.shift(1))[0]
    client_id = 0
    for client_start, client_end in zip(client_bounds[:-1], client_bounds[1:]):
        client_df = purchases.iloc[client_start:client_end]
        trb = np.where(client_df.transaction_id != client_df.transaction_id.shift(1))[0]
        train = client_df[:trb[-1]]
        test = client_df[trb[-1]:]
        total_transactions = train.transaction_id.nunique()
        train['tid'] = train.groupby('transaction_id').cumcount()

        test_products = set(test.product_id.values)
        for product, part in train.groupby('product_id'):
            features['client_id'].append(client_id)
            features['product_id'].append(product)
            features['count'].append(len(part))
            features['count_n'].append(len(part) / total_transactions)
            features['last_transaction'].append(train['tid'].max() - part['tid'].max())
            target.append(product in test_products)

        client_id += 1

    return features, target


def create_features_from_transactions(user_data, products_data):
    trs = user_data['transaction_history']
    products = list(chain(*(x['products'] for x in trs)))
    transaction_ids = list(
        chain(*([i + 1 for _ in range(len(x['products']))] for i, x in enumerate(trs)))
    )
    for i, tid in enumerate(transaction_ids):
        products[i]['tid'] = tid

    total_transactions = len(trs)
    average_psum = sum([tr['purchase_sum'] for tr in trs]) / total_transactions

    features = {
        'total_pucrhases': [],
        'average_psum': [],
        'client_id': [],
        'product_id': [],
        'count': [],
        'count_n': [],
        'last_transaction': [],
    }

    key = itemgetter('product_id')
    for product, part in groupby(sorted(products, key=key), key=key):
        part = list(part)
        features['total_pucrhases'].append(total_transactions)
        features['average_psum'].append(average_psum)
        features['client_id'].append(0)
        features['product_id'].append(product)
        features['count'].append(len(part))
        features['count_n'].append(len(part) / max(transaction_ids))
        features['last_transaction'].append(max([p['tid'] for p in part]) / max(transaction_ids))

    return pd.DataFrame(features).merge(products_data, how='left').fillna(0)
