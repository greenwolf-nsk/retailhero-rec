import pandas as pd
import catboost as cb

from lib.config import TrainConfig
from lib.logger import configure_logger
from lib.recommender import cols, cat_cols

logger = configure_logger(logger_name='train', log_dir='logs')


def train(
        config: TrainConfig,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        products_enriched: pd.DataFrame,
        train_gt_items_count: pd.DataFrame,
        test_gt_items_count: pd.DataFrame,
):

    train_features = pd.merge(train_features, products_enriched, how='left')
    test_features = pd.merge(test_features, products_enriched, how='left')

    columns_diff = set(train_features.columns) - set(cols)
    logger.info(f'columns not used: {columns_diff}')
    for df in (train_features, test_features):
        df['target'] = df['target'].fillna(0).astype(int)
        df.segment_id = df.segment_id.fillna(0).astype(int)
        for col in cat_cols:
            df[col] = df[col].fillna(0)

    median_client_id = train_features.iloc[train_features.shape[0] // 4 * 3].client_id
    split_idx = train_features[train_features['client_id'] == median_client_id].index.max() + 1

    train_df = train_features[:split_idx]
    val_df = train_features[split_idx:]
    test_df = test_features

    groups = {}
    for name, df in (('train', train_df), ('val', val_df), ('test', test_df)):
        client_id_map = {client_id: i for i, client_id in enumerate(df['client_id'].unique())}
        groups[name] = df['client_id'].map(client_id_map).values

    train_pool = cb.Pool(train_df[cols], train_df['target'], cat_features=cat_cols, group_id=groups['train'])
    val_pool = cb.Pool(val_df[cols], val_df['target'], cat_features=cat_cols, group_id=groups['val'])
    test_pool = cb.Pool(test_df[cols], test_df['target'], cat_features=cat_cols, group_id=groups['test'])

    model = cb.CatBoost(config.catboost.train_params)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

    model.save_model(config.catboost.model_file)

    # validate
    gt_cnt_map = {
        'val': train_gt_items_count,
        'test': test_gt_items_count,
    }
    for name, df, pool in (('val', val_df, val_pool), ('test', test_df, test_pool)):
        gt_items_cnt = gt_cnt_map[name]
        df['score'] = model.predict(pool)
        for order in ('score', 'target'):
            scoring = (
                df[['client_id', 'product_id', 'score', 'target']]
                .merge(gt_items_cnt)
                .sort_values(['client_id', order], ascending=[True, False])
            )

            scoring['rank'] = scoring.groupby('client_id').cumcount() + 1
            scoring['cum_target'] = scoring.groupby('client_id')['target'].cumsum()
            scoring['prec'] = ((scoring['rank'] <= 30) * scoring['target'] * (
                        scoring['cum_target'] / scoring['rank']) / scoring['gt_count']).fillna(0)

            score = scoring.groupby('client_id').prec.sum().fillna(0).mean()
            logger.info(f'[{name}] order by {order} : {score}')
