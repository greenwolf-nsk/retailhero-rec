import os
import time

import pandas as pd
import catboost
import flask as fl
from flask import Flask, jsonify

from hardcode import TOP_ITEMS, MAX_RECS
from preprocessing import create_features_from_transactions
from utils import deduplicate


print(time.time())
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.products_data = pd.read_csv(f"{ROOT_DIR}/data/products_enriched.csv")
app.products_data.segment_id = app.products_data.segment_id.fillna(0).astype(int)
app.model = catboost.CatBoost()
app.model.load_model(f"{ROOT_DIR}/models/catboost_rank_gpu.cb")
print(time.time())

cols = [
    'total_pucrhases', 'average_psum', 'count', 'count_n',
    'last_transaction', 'last_transaction_age', 'last_product_transaction_age',
    'level_1', 'level_2', 'level_3', 'level_4', 'segment_id', 'brand_id', 'vendor_id',
    'netto', 'is_own_trademark', 'is_alcohol', 'std', 'mean', 'len'
]
cat_cols = ['level_1', 'level_2', 'level_3', 'level_4', 'segment_id', 'brand_id', 'vendor_id']


@app.route("/ready")
def ready():
    return "OK"


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        features = create_features_from_transactions([fl.request.json]).merge(app.products_data, how='left').fillna(0)
        features.segment_id = features.segment_id.astype(int)
        #cores = app.model.predict_proba(features[cols])[:, 1]
        scores = app.model.predict(features[cols])
        recs = sorted(zip(features["product_id"], scores), key=lambda x: -x[1])
        recs = [x[0] for x in recs]
    except Exception as e:
        print(e)
        recs = []

    return jsonify({"recommended_products": deduplicate(recs + TOP_ITEMS)[:MAX_RECS]})


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=8000)
