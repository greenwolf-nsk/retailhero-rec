from collections import Callable
from operator import itemgetter

import pandas as pd
from catboost import CatBoost

from lib.hardcode import TOP_ITEMS
from lib.i2i_model import ImplicitRecommender
from lib.preprocessing import create_features_from_transactions
from lib.product_store_features import ProductStoreStats
from lib.utils import deduplicate, inplace_hash_join

cols = [
    'total_pucrhases', 'average_psum', 'count', 'p_tr_share', 'last_transaction',
    'last_transaction_age',
    'last_product_transaction_age', 'client_product_dot',
    'level_1', 'level_2', 'level_3', 'level_4',
    'segment_id', 'brand_id', 'vendor_id',
    'netto', 'is_own_trademark', 'is_alcohol', 'implicit_score',
    'first_transaction_age', 'tr_count', 'first_transaction', 'first_product_transaction_age',
    'max_dt', 'min_dt', 'avg_dt',
    'max_q', 'min_q', 'avg_q',
    'max_p', 'min_p', 'avg_p',
    'max_tr_size', 'min_tr_size', 'avg_tr_size',
    'unique_clients_n', 'product_count', 'unique_stores',
    'age', 'gender'
]

cat_cols = [
    'gender',
    'level_1', 'level_2', 'level_3', 'level_4',
    'segment_id', 'brand_id', 'vendor_id',
]


class CatBoostRecommenderWithPopularFallback:

    def __init__(
        self,
        model: CatBoost,
        implicit_model: ImplicitRecommender,
        feature_names: list,
        item_vectors: dict,
        products_data: dict,
        product_store_stats: ProductStoreStats,
    ):
        self.model = model
        self.implicit_model = implicit_model
        self.item_vectors = item_vectors
        self.products_data = products_data
        self.feature_names = feature_names
        self.product_store_stats = product_store_stats

    def recommend(self, user_transactions: dict, limit: int = 30) -> list:
        features = create_features_from_transactions(
            [user_transactions],
            self.item_vectors,
            self.implicit_model,
            self.product_store_stats,
        )
        inplace_hash_join(features, self.products_data)
        features = pd.DataFrame(features)
        features['age'] = user_transactions.get('age', 30)
        features['gender'] = user_transactions.get('gender', 'U')
        for col in cat_cols:
            features[col] = features[col].fillna(0)
        features.segment_id = features.segment_id.astype(int)
        scores = self.model.predict(features[self.feature_names])
        recs = sorted(zip(features['product_id'], scores), key=itemgetter(1), reverse=True)
        product_ids = [rec[0] for rec in recs]

        return deduplicate(product_ids + TOP_ITEMS)[:limit]

    def validate(self, user_transactions: dict, gt_items: list, metric_fn: Callable) -> float:
        recs = self.recommend(user_transactions)
        return metric_fn(gt_items, recs)


lgb_cols = [
    'total_pucrhases', 'average_psum', 'count', 'p_tr_share', 'last_transaction',
    'last_transaction_age',
    'last_product_transaction_age', 'client_product_dot',
    'level_1', 'level_2', 'level_3', 'level_4',
    'segment_id', 'brand_id', 'vendor_id',
    'netto', 'is_own_trademark', 'is_alcohol', 'implicit_score',

    # 'fav_store_count', 'last_store_count',
    # 'fav_product_store_share', 'last_product_store_share',
    # 'fav_store_product_share', 'last_store_product_share',
    'max_dt', 'min_dt', 'avg_dt',
    'max_q', 'min_q', 'avg_q',
    'max_p', 'min_p', 'avg_p',
    'max_tr_size', 'min_tr_size', 'avg_tr_size',
    'unique_clients_n', 'product_count', 'unique_stores',
    # 'product_id',
    #'age', 'gender'
]


class LGBMRecommender:

    def __init__(
        self,
        model,
        implicit_model: ImplicitRecommender,
        feature_names: list,
        item_vectors: dict,
        products_data: dict,
        product_store_stats: ProductStoreStats,
    ):
        self.model = model
        self.implicit_model = implicit_model
        self.item_vectors = item_vectors
        self.products_data = products_data
        self.feature_names = feature_names
        self.product_store_stats = product_store_stats

    def recommend(self, user_transactions: dict, limit: int = 30) -> list:
        features = create_features_from_transactions(
            [user_transactions],
            self.item_vectors,
            self.implicit_model,
            self.product_store_stats,
        )
        inplace_hash_join(features, self.products_data)
        features = pd.DataFrame(features)
        for col in cat_cols[1:]:
            features[col] = features[col].fillna(0)
        features.segment_id = features.segment_id.astype(int)
        scores = self.model.predict(features[self.feature_names])
        recs = sorted(zip(features['product_id'], scores), key=itemgetter(1), reverse=True)
        product_ids = [rec[0] for rec in recs]

        return deduplicate(product_ids + TOP_ITEMS)[:limit]

    def validate(self, user_transactions: dict, gt_items: list, metric_fn: Callable) -> float:
        recs = self.recommend(user_transactions)
        return metric_fn(gt_items, recs)
