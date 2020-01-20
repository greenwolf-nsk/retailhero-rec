from collections import Counter


class PopularHistoryRecommender:

    @staticmethod
    def recommend(user_history: dict):
        counter = Counter()
        purchases = user_history['transaction_history']
        for record in purchases:
            product_ids = [product['product_id'] for product in record['products']]
            counter.update(product_ids)

        return [x[0] for x in counter.most_common(30)]


class CoocRecommender:

    def __init__(self, cooc_dict):
        self.cooc_dict = cooc_dict

    def recommend(self, user_history: dict):
        counter = Counter()
        purchases = user_history['transaction_history']
        for record in purchases:
            for product_id in [product['product_id'] for product in record['products']]:
                counter.update(self.cooc_dict[product_id])

        return [x[0] for x in counter.most_common(30)]
