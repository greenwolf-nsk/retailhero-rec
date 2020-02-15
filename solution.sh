FNAME=$1

zip -r $FNAME lib server.py metadata.json models/catboost_rank_dot_200k.cb data/products_enriched_200k.csv data/item_vectors_collab.json configs