from collections import Callable
from operator import itemgetter

import pandas as pd
from catboost import CatBoost

from lib.hardcode import TOP_ITEMS
from lib.preprocessing import create_features_from_transactions
from lib.utils import deduplicate


cols = [
    'total_pucrhases', 'average_psum', 'count', 'p_tr_share', 'last_transaction',
    'last_transaction_age', 'last_product_transaction_age', 'client_product_dot',
    'level_1', 'level_2', 'level_3', 'level_4', 'segment_id', 'brand_id', 'vendor_id',
    'netto', 'is_own_trademark', 'is_alcohol',
    'max_dt', 'min_dt', 'avg_dt', 'max_q', 'min_q', 'avg_q', 'unique_clients'
]


class CatBoostRecommenderWithPopularFallback:

    def __init__(
        self,
        model: CatBoost,
        feature_names: list,
        item_vectors: dict,
        products_data: pd.DataFrame,
    ):
        self.model = model
        self.item_vectors = item_vectors
        self.products_data = products_data
        self.feature_names = feature_names

    def recommend(self, user_transactions: dict, limit: int = 30) -> list:
        features = (
            create_features_from_transactions([user_transactions], self.item_vectors)
            .merge(self.products_data, how='left')
            .fillna(0)
        )
        features.segment_id = features.segment_id.astype(int)
        scores = self.model.predict(features[self.feature_names])
        recs = sorted(zip(features['product_id'], scores), key=itemgetter(1), reverse=True)
        product_ids = [rec[0] for rec in recs]

        return deduplicate(product_ids + TOP_ITEMS)[:limit]

    def recommend_with_dot_only(self, user_transactions: dict, limit: int = 30) -> list:
        features = (
            create_features_from_transactions([user_transactions], self.item_vectors)
            .merge(self.products_data, how='left')
            .fillna(0)
        )
        recs = sorted(
            zip(features['product_id'], features['client_product_dot']),
            key=itemgetter(1),
            reverse=True
        )
        product_ids = [rec[0] for rec in recs]

        return deduplicate(product_ids + TOP_ITEMS)[:limit]

    def validate(self, user_transactions: dict, gt_items: list, metric_fn: Callable) -> float:
        recs = self.recommend(user_transactions)
        return metric_fn(gt_items, recs)

    def validate_dot(self, user_transactions: dict, gt_items: list, metric_fn: Callable) -> float:
        recs = self.recommend_with_dot_only(user_transactions)
        return metric_fn(gt_items, recs)
