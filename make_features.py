import json
import pickle
import sys

import implicit
import pandas as pd

from lib.config import TrainConfig, ImplicitConfig
from lib.logger import configure_logger
from lib.train_utils import read_clients_purchases
from lib.i2i_model import create_sparse_purchases_matrix, ProductIdMap
from lib.preprocessing import (
    create_features_from_transactions,
    create_product_features_from_users_data,
    create_target_from_transactions,
    create_gt_items_count_df
)

logger = configure_logger(logger_name='make_features', log_dir='logs')


def train_implicit_vectors(
        train_records: list,
        config: ImplicitConfig,
        product_id_map: ProductIdMap
):
    matrix = create_sparse_purchases_matrix(train_records, product_id_map)
    model = implicit.als.AlternatingLeastSquares(
        factors=config.num_factors,
        iterations=config.epochs
    )
    model.fit(matrix)
    item_vectors = {
        product_id_map.to_product(i): list(map(float, factor))
        for i, factor in enumerate(model.user_factors)  # user factors, cuz in implicit its inverted
    }
    with open(config.vectors_file, 'w') as f:
        json.dump(item_vectors, f)

    return item_vectors


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = TrainConfig.from_json(config_path)

    logger.info(f'config: {config_path}')
    train_records, test_records = read_clients_purchases(
        config.client_purchases_file,
        config.client_offset,
        config.client_limit
    )
    logger.info(f'read {config.client_limit} clients purchases from {config.client_offset}')
    with open(config.implicit.model_file, 'rb') as f:
        model = pickle.load(f)
    item_vectors = train_implicit_vectors(train_records, config.implicit, model.product_id_map)

    logger.info(f'trained vectors for {len(item_vectors)} items')

    features = create_features_from_transactions(train_records, item_vectors, model)
    target = create_target_from_transactions(test_records)
    gt_items_count = create_gt_items_count_df(target)
    features_df = features.merge(target, how='left', sort=False)
    features_df.to_csv(config.features_file, index=False)
    gt_items_count.to_csv(config.gt_items_count_file, index=False)
    logger.info(f'created features and target, shape: {features.shape}')

    product_features = create_product_features_from_users_data(train_records)
    products = pd.read_csv(config.products_file)
    products_enriched = pd.merge(products, product_features, how='left')
    products_enriched.to_csv(config.products_enriched_file, index=False)
    logger.info(f'created enriched product features, shape: {products_enriched.shape}')
