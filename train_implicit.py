import sys
import pickle

import implicit
import pandas as pd

from lib.config import TrainConfig
from lib.logger import configure_logger
from lib.train_utils import read_clients_purchases
from lib.i2i_model import create_sparse_purchases_matrix, ImplicitRecommender, ProductIdMap

logger = configure_logger(logger_name='train_implicit', log_dir='logs')

if __name__ == '__main__':
    config_path = sys.argv[1]
    config = TrainConfig.from_json(config_path)

    products = pd.read_csv(config.products_file)
    product_id_map = ProductIdMap(products['product_id'].values)

    logger.info(f'read {config.client_limit} clients purchases from {config.client_offset}')
    train_records, test_recrods = read_clients_purchases(
        config.client_purchases_file,
        config.client_offset, config.client_limit
    )
    model = implicit.nearest_neighbours.ItemItemRecommender(K=10)
    matrix = create_sparse_purchases_matrix(train_records, product_id_map)
    model.fit(matrix.T)

    recommender = ImplicitRecommender(model, product_id_map)
    with open(config.implicit.model_file, 'wb') as f:
        pickle.dump(recommender, f)

    logger.info(f'saved model to {config.implicit.model_file}')
