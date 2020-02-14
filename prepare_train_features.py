import pandas as pd

from i2i_data import load_item_vectors
from train_utils import read_clients_purchases
from solution.preprocessing import (
    create_features_from_transactions,
    create_target_from_transactions,
    create_product_features_from_users_data,
)


products_data = pd.read_csv('../data/products_enriched.csv')
item_vectors = load_item_vectors('../data/item_vectors.json')

train, test = read_clients_purchases('../data/clients_purchases.tsv', offset=1000, limit=100000)

features = create_features_from_transactions(train, item_vectors)

target = create_target_from_transactions(test)

product_features = create_product_features_from_users_data(train)

features_df = features.merge(target, how='left', sort=False)

features_df.head()

features_df.to_csv('../data/features.csv', index=False)

product_features.to_csv('../data/additional_product_features.csv', index=False)