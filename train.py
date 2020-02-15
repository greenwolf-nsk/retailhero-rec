import sys

import pandas as pd
import catboost as cb

from lib.config import TrainConfig
from lib.logger import configure_logger
from lib.recommender import cols, cat_cols

logger = configure_logger(logger_name='train', log_dir='logs')

if __name__ == '__main__':
    config_path = sys.argv[1]
    config = TrainConfig.from_json(config_path)
    logger.info(f'config: {config_path}')

    features = pd.read_csv(config.features_file)
    products_enriched = pd.read_csv(config.products_enriched_file)

    features = pd.merge(features, products_enriched, how='left').fillna(0)
    features['target'] = features['target'].fillna(0).astype(int)
    features.segment_id = features.segment_id.astype(int)

    median_client_id = features.iloc[features.shape[0] // 4 * 3].client_id
    split_idx = features[features['client_id'] == median_client_id].index.max() + 1

    df = features[:split_idx]
    test_df = features[split_idx:]

    client_id_map = {client_id: i for i, client_id in enumerate(df['client_id'].unique())}
    tclient_id_map = {client_id: i for i, client_id in enumerate(test_df['client_id'].unique())}

    train_groups = df['client_id'].map(client_id_map).values
    test_groups = test_df['client_id'].map(tclient_id_map).values

    train_pool = cb.Pool(df[cols], df['target'], cat_features=cat_cols, group_id=train_groups)
    test_pool = cb.Pool(test_df[cols], test_df['target'], cat_features=cat_cols, group_id=test_groups)

    model = cb.CatBoost(config.catboost.train_params)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100)

    model.save_model(config.catboost.model_file)

    # validate
    client_gt_items_cnt = pd.read_csv(config.gt_items_count_file)
    test_df['score'] = model.predict(test_pool)
    scoring = (
        test_df[['client_id', 'product_id', 'score', 'target']]
        .merge(client_gt_items_cnt)
        .sort_values(['client_id', 'score'], ascending=[True, False])
    )

    scoring['rank'] = scoring.groupby('client_id').cumcount() + 1
    scoring['cum_target'] = scoring.groupby('client_id')['target'].cumsum()
    scoring['prec'] = ((scoring['rank'] <= 30) * scoring['target'] * (
                scoring['cum_target'] / scoring['rank']) / scoring['gt_count']).fillna(0)

    score = scoring.groupby('client_id').prec.sum().fillna(0).mean()
    logger.info(score)




