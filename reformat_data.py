import json
from pathlib import Path
from typing import List

import attr
import tqdm as tqdm


@attr.s(slots=True)
class PurchaseRow:
    client_id = attr.ib()
    transaction_id = attr.ib()
    transaction_datetime = attr.ib()
    regular_points_received = attr.ib(converter=float)
    express_points_received = attr.ib(converter=float)
    regular_points_spent = attr.ib(converter=float)
    express_points_spent = attr.ib(converter=float)
    purchase_sum = attr.ib(converter=float)
    store_id = attr.ib()
    product_id = attr.ib()
    product_quantity = attr.ib(converter=float)
    trn_sum_from_iss = attr.ib(converter=float)
    trn_sum_from_red = attr.ib()


def parse_purchases(purchases_fp: Path, out_fp: Path, split_date: str):
    clients_total = 0
    clients_test = 0
    with open(out_fp, 'w') as fout:
        with open(purchases_fp, 'r') as fin:
            fin.readline()
            first_row = PurchaseRow(*fin.readline().strip().split(','))
            current_client_id = first_row.client_id
            client_purchases_train = [first_row]
            client_purchases_test = []
            for line in tqdm.tqdm(fin):
                row = PurchaseRow(*line.strip().split(','))
                if current_client_id != row.client_id:
                    train = json.dumps(process_client_purchases(client_purchases_train))
                    clients_total += 1
                    if client_purchases_test:
                        # skip records w/o actions in test period
                        test = json.dumps(process_client_purchases(client_purchases_test))
                        fout.write(f'{train}\t{test}\n')
                        clients_test += 1
                    client_purchases_train = [row]
                    client_purchases_test = []
                    current_client_id = row.client_id
                else:
                    if row.transaction_datetime < split_date:
                        client_purchases_train.append(row)
                    else:
                        client_purchases_test.append(row)

    print(f'total: {clients_total}, test: {clients_test}')


def process_client_purchases(client_purchases: List[PurchaseRow]) -> dict:
    first_row = client_purchases[0]
    client_record = {
        'client_id': first_row.client_id
    }
    transactions = []
    current_transaction = {}
    current_transaction_id = -1
    for row in client_purchases[1:]:
        if row.transaction_id != current_transaction_id:
            current_transaction = {
                'datetime': row.transaction_datetime,
                'purchase_sum': row.purchase_sum,
                'store_id': row.store_id,
                'products': [{
                    'product_id': row.product_id,
                    'price': row.trn_sum_from_iss,
                    'quantity': row.product_quantity,
                }]
            }
            if current_transaction_id != -1:
                transactions.append(current_transaction)
            current_transaction_id = row.transaction_id
        else:
            current_transaction['products'].append({
                'product_id': row.product_id,
                'price': row.trn_sum_from_iss,
                'quantity': row.product_quantity,
            })

    client_record['transaction_history'] = transactions
    return client_record


if __name__ == '__main__':
    split_date = '2019-03-01 00:00:00'
    parse_purchases('./data/purchases.csv', './data/clients_purchases.tsv', split_date)
