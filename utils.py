import json
from collections import namedtuple

from pathlib import Path


Query = namedtuple('Query', 'history answers')


def parse_check_queries_file(filename: Path):
    with open(filename, 'r') as f:
        lines = f.readlines()

    queries = []
    for line in lines:
        history, answer = line.split('\t')
        queries.append(Query(json.loads(history), json.loads(answer)))

    return queries

