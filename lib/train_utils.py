import json
from pathlib import Path


def read_clients_purchases(fp: Path, offset: int = 0, limit: int = 1000) -> list:
    train_records = []
    test_records = []
    with open(fp, 'r') as f:
        for line_no, line in enumerate(f):
            if line_no < offset:
                continue
            elif line_no < offset + limit:
                train, test = line.strip().split('\t')
                train, test = json.loads(train), json.loads(test)
                train_records.append(train)
                test_records.append(test)

    return train_records, test_records
