import json
from pathlib import Path


def read_clients_purchases(fp: Path, start: int = 0, end: int = 1000) -> list:
    train_records = []
    test_records = []
    with open(fp, 'r') as f:
        for line_no, line in enumerate(f):
            if line_no < start:
                continue
            elif line_no < end:
                train, test = line.strip().split('\t')
                train, test = json.loads(train), json.loads(test)
                train_records.append(train)
                test_records.append(test)

    return train_records, test_records
