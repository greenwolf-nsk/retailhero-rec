import argparse
from warnings import filterwarnings
filterwarnings('ignore')

import implicit
import pandas as pd

from lib.config import TrainConfig
from lib.logger import configure_logger
from lib.product_store_features import create_product_store_stats, ProductStoreStats
from lib.train_utils import read_clients_purchases
from lib.i2i_model import create_sparse_purchases_matrix, ProductIdMap, ImplicitRecommender, \
    train_implicit_vectors
from lib.preprocessing import (
    create_features_from_transactions,
    create_product_features_from_users_data,
    create_target_from_transactions,
    create_gt_items_count_df
)
from lib.utils import pickle_dump, pickle_load
from train import train

logger = configure_logger(logger_name='make_features', log_dir='logs')


def create_features(
        seed_records: list,
        target_records: list,
        item_vectors: dict,
        recommender: ImplicitRecommender,
        product_store_stats: ProductStoreStats,
):
    features_dict = create_features_from_transactions(
        seed_records,
        item_vectors,
        recommender,
        product_store_stats
    )
    features = pd.DataFrame(features_dict)
    target = create_target_from_transactions(target_records)
    gt_items_count = create_gt_items_count_df(target)
    features_df = features.merge(target, how='left', sort=False)
    return features_df, gt_items_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path')
    parser.add_argument('--train_implicit_model', action='store_true')
    parser.add_argument('--train_vectors', action='store_true')
    parser.add_argument('--calc_product_store_stats', action='store_true')
    parser.add_argument('--calc_product_features', action='store_true')
    parser.add_argument('--create_train_features', action='store_true')
    parser.add_argument('--create_test_features', action='store_true')
    parser.add_argument('--train_model', action='store_true')
    parser.add_argument('--skip_data_read', action='store_true')
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config_path)

    logger.info(f'config: {args.config_path}')

    if not args.skip_data_read:
        train_seed_records, train_target_recrods = read_clients_purchases(
            config.client_purchases_file,
            config.train_start,
            config.train_end
        )
        logger.info(f'train: {config.train_start} - {config.train_end}')
        test_seed_records, test_target_records = read_clients_purchases(
            config.client_purchases_file,
            config.test_start,
            config.test_end
        )
        logger.info(f'test: {config.test_start} - {config.test_end}')

        products = pd.read_csv(config.products_file)
        product_id_map = ProductIdMap(products['product_id'].values)

    if args.train_implicit_model:
        logger.info('training implicit model...')
        model = implicit.nearest_neighbours.ItemItemRecommender(K=10)
        matrix = create_sparse_purchases_matrix(train_seed_records, product_id_map)
        model.fit(matrix.T)
        recommender = ImplicitRecommender(model, product_id_map)
        pickle_dump(config.implicit.model_file, recommender)
        logger.info(f'saved model to {config.implicit.model_file}')
    else:
        logger.info(f'loading implicit model from file')
        recommender = pickle_load(config.implicit.model_file)

    if args.train_vectors:
        logger.info(f'training vectors...')
        item_vectors = train_implicit_vectors(train_seed_records, config.implicit, product_id_map)
        pickle_dump(config.implicit.vectors_file, item_vectors)
        logger.info(f'trained vectors for {len(item_vectors)} items')
    else:
        logger.info(f'loading vectors from file')
        item_vectors = pickle_load(config.implicit.vectors_file)

    if args.calc_product_store_stats:
        logger.info(f'calculating product/store stats...')
        product_store_stats = create_product_store_stats(train_seed_records)
        pickle_dump(config.product_store_stats_file, product_store_stats)
        logger.info(f'calculated and saved product store stats')
    else:
        logger.info(f'loading product/store stats from file')
        product_store_stats = pickle_load(config.product_store_stats_file)

    if args.calc_product_features:
        logger.info(f'calculating product features stats...')
        product_features = create_product_features_from_users_data(train_seed_records)
        products = pd.read_csv(config.products_file)
        products_enriched = pd.merge(products, product_features, how='left')
        products_enriched.to_csv(config.products_enriched_file, index=False)
        logger.info(f'created enriched product features, shape: {products_enriched.shape}')
    else:
        logger.info(f'reading product stats from file...')
        products_enriched = pd.read_csv(config.products_enriched_file)

    if args.create_train_features:
        logger.info(f'creating train features...')
        train_features_df, train_gt_items_count = create_features(
            train_seed_records,
            train_target_recrods,
            item_vectors,
            recommender,
            product_store_stats,
        )
        train_features_df.to_csv(config.train_features_file, index=False)
        train_gt_items_count.to_csv(config.train_gt_items_count_file, index=False)
        logger.info(f'created train features and target, shape: {train_features_df.shape}')
    else:
        logger.info(f'reading train features from file...')
        train_features_df = pd.read_csv(config.train_features_file)
        train_gt_items_count = pd.read_csv(config.train_gt_items_count_file)
        logger.info(f'read train features and target, shape: {train_features_df.shape}')

    if args.create_test_features:
        logger.info(f'creating test features...')
        test_features_df, test_gt_items_count = create_features(
            test_seed_records,
            test_target_records,
            item_vectors,
            recommender,
            product_store_stats,
        )
        test_features_df.to_csv(config.test_features_file, index=False)
        test_gt_items_count.to_csv(config.test_gt_items_count_file, index=False)
        logger.info(f'created test features and target, shape: {test_features_df.shape}')
    else:
        logger.info(f'reading test features from file...')
        test_features_df = pd.read_csv(config.test_features_file)
        test_gt_items_count = pd.read_csv(config.test_gt_items_count_file)
        logger.info(f'read test features and target, shape: {test_features_df.shape}')

    clients_df = pd.read_csv('./data/clients.csv')
    test_features_df = test_features_df.merge(clients_df, how='left')
    train_features_df = train_features_df.merge(clients_df, how='left')

    if args.train_model:
        logger.info(f'training model...')
        train(
            config,
            train_features_df,
            test_features_df,
            products_enriched,
            train_gt_items_count,
            test_gt_items_count,
        )


