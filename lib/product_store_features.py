from collections import defaultdict, Counter
from operator import itemgetter


def ddf():
    return defaultdict(int)


class ProductStoreStats:

    def __init__(self):
        self.store_product_count = defaultdict(ddf)
        self.store_count = defaultdict(int)
        self.product_count = defaultdict(int)
        self.product_id_map = {}
        self.store_id_map = {}

    def add(self, product_id: str, store_id: str):
        self.store_product_count[store_id][product_id] += 1
        self.store_count[store_id] += 1
        self.product_count[product_id] += 1

    def optimize(self):
        self.product_id_map = {pid: i for i, pid in enumerate(self.product_count)}
        self.store_id_map = {sid: i for i, sid in enumerate(self.store_count)}
        self.product_count = {
            self.product_id_map[pid]: cnt
            for pid, cnt in self.product_count.items()
        }
        self.store_count = {
            self.store_id_map[sid]: cnt
            for sid, cnt in self.store_count.items()
        }
        self.store_product_count = {
            self.store_id_map[sid]: {
                self.product_id_map[pid]: cnt
                for pid, cnt in product.items()
            }
            for sid, product in self.store_product_count.items()
        }

    def store_cnt(self, store_id: str) -> int:
        if store_id in self.store_id_map:
            store_id = self.store_id_map[store_id]
            return self.store_count.get(store_id, 0)
        return 0

    def product_store_share(self, product_id: str, store_id: str) -> float:
        try:
            if store_id in self.store_id_map and product_id in self.product_id_map:
                product_id, store_id = self.product_id_map[product_id], self.store_id_map[store_id]
                if product_id in self.product_count:
                    return self.store_product_count[store_id][product_id] / self.product_count[product_id]
        except:
            return 0

    def store_product_share(self, product_id: str, store_id: str) -> float:
        try:
            if store_id in self.store_id_map and product_id in self.product_id_map:
                product_id, store_id = self.product_id_map[product_id], self.store_id_map[store_id]
                if store_id in self.store_count:
                    return self.store_product_count[store_id][product_id] / self.store_count[store_id]
        except:
            return 0


def create_product_store_stats(users_data: list) -> ProductStoreStats:
    stats = ProductStoreStats()
    for record in users_data:
        for tr in record['transaction_history']:
            store_id = tr['store_id']
            for product in tr['products']:
                product_id = product['product_id']
                stats.add(product_id, store_id)

    stats.optimize()

    return stats


def get_user_favorite_store(user_record: dict) -> str:
    store_ids = [
        tr['store_id']
        for tr in user_record['transaction_history']
    ]
    return Counter(store_ids).most_common(1)[0][0]


def get_user_last_store(user_record: dict) -> str:
    store_ids = [
        tr['store_id']
        for tr in user_record['transaction_history']
    ]
    return store_ids[-1]
