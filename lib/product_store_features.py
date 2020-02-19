from collections import defaultdict, Counter


def ddf():
    return defaultdict(int)


class ProductStoreStats:

    def __init__(self):
        self.product_store_count = defaultdict(ddf)
        self.store_count = defaultdict(int)
        self.product_count = defaultdict(int)

    def add(self, product_id: str, store_id: str):
        self.product_store_count[product_id][store_id] += 1
        self.store_count[store_id] += 1
        self.product_count[product_id] += 1

    def product_store_share(self, product_id: str, store_id: str):
        if self.product_count[product_id]:
            return self.product_store_count[product_id][store_id] / self.product_count[product_id]
        else:
            return 0

    def store_product_share(self, product_id: str, store_id: str):
        if self.store_count[store_id]:
            return self.product_store_count[product_id][store_id] / self.store_count[store_id]
        else:
            return 0


def create_product_store_stats(users_data: list) -> ProductStoreStats:
    stats = ProductStoreStats()
    for record in users_data:
        for tr in record['transaction_history']:
            store_id = tr['store_id']
            for product in tr['products']:
                product_id = product['product_id']
                stats.add(product_id, store_id)

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
