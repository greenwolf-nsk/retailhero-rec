FROM python:3.7-stretch
RUN python3 -m pip install -U scikit-learn pandas scipy numpy gunicorn flask
RUN python3 -m pip install -U pandas catboost