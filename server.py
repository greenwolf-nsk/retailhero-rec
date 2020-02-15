import os
import time

import pandas as pd
import catboost
import flask as fl
from flask import Flask, jsonify

from lib.hardcode import TOP_ITEMS
from lib.i2i_data import load_item_vectors
from lib.recommender import CatBoostRecommenderWithPopularFallback, cols

print(time.time())
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.products_data = pd.read_csv(f"{ROOT_DIR}/data/products_enriched_200k.csv")
app.products_data.segment_id = app.products_data.segment_id.fillna(0).astype(int)
app.model = catboost.CatBoost()
app.model.load_model(f"{ROOT_DIR}/models/catboost_rank_dot_200k.cb")
app.item_vectors = load_item_vectors(f"{ROOT_DIR}/data/item_vectors_collab.json")
app.recommender = CatBoostRecommenderWithPopularFallback(
    model=app.model,
    item_vectors=app.item_vectors,
    products_data=app.products_data,
    feature_names=cols,

)
print(time.time())


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
