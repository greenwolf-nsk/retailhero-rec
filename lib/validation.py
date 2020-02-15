from run_queries import normalized_average_precision
from lib.recommender import CatBoostRecommenderWithPopularFallback


def validate(
        recommender: CatBoostRecommenderWithPopularFallback,
        history: list,
        dot_only: bool = False,
):
    scores = []
    for user_seed, test_data in history:
        if test_data['transaction_history']:
            next_transaction = test_data['transaction_history'][0]
            gt_items = [p['product_id'] for p in next_transaction['products']]
            if dot_only:
                score = recommender.validate_dot(user_seed, gt_items, normalized_average_precision)
            else:
                score = recommender.validate(user_seed, gt_items, normalized_average_precision)

            scores.append(score)

    print(sum(scores) / len(scores))
