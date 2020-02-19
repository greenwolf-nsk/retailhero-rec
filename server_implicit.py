import os
import pickle

import flask as fl
from flask import Flask, jsonify

from lib.config import TrainConfig
from lib.hardcode import TOP_ITEMS
from lib.logger import configure_logger

logger = configure_logger(logger_name='server', log_dir='')


logger.info('starting to load all stuff')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
config = TrainConfig.from_json('configs/train_config_300k.json')

app = Flask(__name__)
with open(config.implicit.model_file, 'rb') as f:
    app.recommender = pickle.load(f)

logger.info('ready!')


@app.route("/ready")
def ready():
    return "OK"


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        recs = [x[0] for x in app.recommender.recommend(fl.request.json)]
    except Exception as e:
        print(e)
        recs = TOP_ITEMS

    return jsonify({"recommended_products": recs})


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host="0.0.0.0", debug=True, port=8000)
