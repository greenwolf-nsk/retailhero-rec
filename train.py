import pandas as pd
import catboost as cb
import numpy as np

features = pd.read_csv('../data/features_200k.csv')
products = pd.read_csv('../data/products.csv')
additional_product_features = pd.read_csv('../data/additional_product_features.csv')

additional_product_features.head()

features = pd.merge(features, products, how='left').fillna(0)
features = pd.merge(features, additional_product_features, how='left').fillna(0)

features.head()

features['target'] = features['target'].fillna(0).astype(int)

features.segment_id = features.segment_id.astype(int)

median_client_id = features.iloc[features.shape[0] // 2].client_id
split_idx = features[features['client_id'] == median_client_id].index.max() + 1

df = features[:split_idx]
test_df = features[split_idx:]

cols = [
    'total_pucrhases', 'average_psum', 'count', 'p_tr_share', 'last_transaction',
    'last_transaction_age', 'last_product_transaction_age', 'client_product_cosine',
    'level_1', 'level_2', 'level_3', 'level_4', 'segment_id', 'brand_id', 'vendor_id',
    'netto', 'is_own_trademark', 'is_alcohol',
    'max_dt', 'min_dt', 'avg_dt', 'max_q', 'min_q', 'avg_q', 'unique_clients'
]
cat_cols = ['level_1', 'level_2', 'level_3', 'level_4', 'segment_id', 'brand_id', 'vendor_id']

client_id_map = {client_id: i for i, client_id in enumerate(df['client_id'].unique())}
tclient_id_map = {client_id: i for i, client_id in enumerate(test_df['client_id'].unique())}

train_groups = df['client_id'].map(client_id_map).values
test_groups = test_df['client_id'].map(tclient_id_map).values

train_pool = cb.Pool(df[cols], df['target'], cat_features=cat_cols, group_id=train_groups)
test_pool = cb.Pool(test_df[cols], test_df['target'], cat_features=cat_cols, group_id=test_groups)

params = {
    'objective': 'Logloss',
    'task_type': 'GPU',
    'eval_metric': 'MAP',
    'iterations': 500,
    'verbose': 10,
}

clf = cb.CatBoost(params)
clf.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100)

clf.save_model('../solution/models/catboost_rank_cosine_200k.cb')