import json
import os
from pathlib import Path


class Config:

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)

    @classmethod
    def from_json(cls, fp: Path):
        with open(fp, 'r') as f:
            return cls(**json.load(f))


class ImplicitConfig(Config):

    def __init__(
        self,
        epochs: int,
        num_factors: int,
        vectors_file: str,
        model_file: str,
    ):
        self.epochs = epochs
        self.num_factors = num_factors
        self.vectors_file = vectors_file
        self.model_file = model_file
        
        
class CatboostConfig(Config):

    def __init__(
        self,
        model_file: int,
        train_params: dict,
    ):
        self.model_file = model_file
        self.train_params = train_params


class TrainConfig(Config):
    def __init__(
        self,
        data_dir: Path,
        log_dir: Path,
        products_file: str,
        train_features_file: str,
        train_gt_items_count_file: str,
        test_features_file: str,
        test_gt_items_count_file: str,
        client_purchases_file: str,
        products_enriched_file: str,
        product_store_stats_file: str,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
        implicit: dict,
        catboost: dict,
    ):
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.products_file = os.path.join(data_dir, products_file)
        self.train_features_file = os.path.join(data_dir, train_features_file)
        self.test_features_file = os.path.join(data_dir, test_features_file)
        self.train_gt_items_count_file = os.path.join(data_dir, train_gt_items_count_file)
        self.test_gt_items_count_file = os.path.join(data_dir, test_gt_items_count_file)
        self.client_purchases_file = os.path.join(data_dir, client_purchases_file)
        self.products_enriched_file = os.path.join(data_dir, products_enriched_file)
        self.product_store_stats_file = os.path.join(data_dir, product_store_stats_file)

        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

        self.implicit = ImplicitConfig.from_dict(implicit)
        self.implicit.vectors_file = os.path.join(data_dir, self.implicit.vectors_file)
        self.implicit.model_file = os.path.join(data_dir, self.implicit.model_file)

        self.catboost = CatboostConfig.from_dict(catboost)
        self.catboost.model_file = os.path.join(data_dir, self.catboost.model_file)
