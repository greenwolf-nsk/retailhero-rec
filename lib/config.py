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
        features_file: str,
        gt_items_count_file: str,
        client_purchases_file: str,
        products_enriched_file: str,
        client_offset: int,
        client_limit: int,
        implicit: dict,
        catboost: dict,
    ):
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.products_file = os.path.join(data_dir, products_file)
        self.features_file = os.path.join(data_dir, features_file)
        self.gt_items_count_file = os.path.join(data_dir, gt_items_count_file)
        self.client_purchases_file = os.path.join(data_dir, client_purchases_file)
        self.products_enriched_file = os.path.join(data_dir, products_enriched_file)

        self.client_offset = client_offset
        self.client_limit = client_limit

        self.implicit = ImplicitConfig.from_dict(implicit)
        self.implicit.vectors_file = os.path.join(data_dir, self.implicit.vectors_file)
        self.implicit.model_file = os.path.join(data_dir, self.implicit.model_file)

        self.catboost = CatboostConfig.from_dict(catboost)
        self.catboost.model_file = os.path.join(data_dir, self.catboost.model_file)


if __name__ == '__main__':
    train_config = {
        'data_dir': '../data',
        'products_file': 'products.csv',
        'products_enriched_file':  'products_enriched.csv',
        'client_purchases_file': 'client_purchases.tsv',
        'client_offset': 1000,
        'client_limit': 200_000,
        'implicit': {
            'epochs': 50,
            'num_factors': 64,
            'vectors_file': 'vectors.json'
        },
        'catboost': {
            'model_file': 'catboost_200k.cb',
            'train_params': {
                'objective': 'Logloss',
                'task_type': 'GPU',
                'eval_metric': 'MAP',
                'iterations': 500,
                'verbose': 10,
            }
        }
    }
    cnf = TrainConfig(**train_config)
    print(cnf.catboost.train_params)

