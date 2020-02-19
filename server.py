import os
import pickle
import time

import pandas as pd
import catboost
import flask as fl
from flask import Flask, jsonify

from lib.config import TrainConfig
from lib.hardcode import TOP_ITEMS
from lib.i2i_model import load_item_vectors
from lib.logger import configure_logger
from lib.recommender import CatBoostRecommenderWithPopularFallback, cols
from lib.utils import read_products_file

logger = configure_logger(logger_name='server', log_dir='')


logger.info('starting to load all stuff')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
config = TrainConfig.from_json('configs/train_config_small.json')

app = Flask(__name__)
app.products_data = read_products_file(config.products_enriched_file)

app.model = catboost.CatBoost()
app.model.load_model(config.catboost.model_file)
app.item_vectors = load_item_vectors(config.implicit.vectors_file)
with open(config.implicit.model_file, 'rb') as f:
    app.implicit_model = pickle.load(f)

with open(config.product_store_stats_file, 'rb') as f:
    app.product_store_stats = pickle.load(f)

app.recommender = CatBoostRecommenderWithPopularFallback(
    model=app.model,
    implicit_model=app.implicit_model,
    item_vectors=app.item_vectors,
    products_data=app.products_data,
    product_store_stats=app.product_store_stats,
    feature_names=cols,
)

logger.info('ready!')


@app.route("/ready")
def ready():
    return "OK"


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        recs = app.recommender.recommend(fl.request.json)
    except Exception as e:
        print(e)
        recs = TOP_ITEMS

    return jsonify({"recommended_products": recs})


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=8000)
