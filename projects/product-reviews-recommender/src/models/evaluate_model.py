import pandas as pd


def recall_at_n(asins: list, predicted_asins: list) -> float:
    """

    Args:
        asins ([list]):
        predicted_asins ([list]):
    """
    # number of relevant items
    set_actual = set(asins)
    set_preds = set(predicted_asins)
    num_relevant = len(set_actual.intersection(set_preds))

    # calculating recall@N - relevant / total relevant items
    recall_at_n = num_relevant / len(asins)

    return recall_at_n


def novelty_at_n(
    item_popularity: pd.DataFrame, predicted_asins: list, n: int = 10
) -> float:
    """

    Args:
        item_popularity ([pd.DataFrame]):
        predicted_asins ([list]):
        k ([int]):
    """
    # finding avg novelty
    popularity_sum = item_popularity.loc[predicted_asins].sum()
    novelty_at_n = ((n * 1) - popularity_sum) / n

    return novelty_at_n


def generate_item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    """Compute item popularity based on item's no. reviews / highest no. reivews given to an item.

    Args:
        train ([pd.DataFrame]): Training dataset for recommender system.
    """

    # create a mapping of item popularatity
    # based on sum(item's review / max reviews) / no items
    max_reviews = (
        train.groupby(["asin"]).agg({"processedReviewText": "count"}).max().values[0]
    )
    item_popularity = (
        train.groupby(["asin"])
        .agg({"processedReviewText": "count"})
        .apply(lambda x: x / max_reviews)
    )

    return item_popularity


def evaluate_recommendations(
    model_name: str,
    top_ns: dict,
    user_rating_history: pd.DataFrame,
    item_popularity: pd.DataFrame,
    n: int = 10,
    mf_based: bool = False,
) -> pd.DataFrame:
    """Compute metrics ``Recall@N`` and ``Novelty@N`` given N-number of recommended items (Top-N items).

    Args:
        model_name ([str]): Learning algorithm for evaluating recommendation.
        top_ns ([dict]): Top-N recommendations.
        user_rating_history ([pd.DataFrame]): Past purchase history aggregated on user level (test data).
        item_popularity ([pd.DataFrame]): DataFrame of the item popularity as defined by no. reviews / highest no. of reviews given to an item.
        n ([int]): Number of recommended items.
        mf_based ([bool]): Whether algorithm is based on Matrix Factorization. Defaults to ``False``.
    """

    test_recommendations = pd.DataFrame(
        top_ns.items(), columns=["reviewerID", "pred_asin"]
    )

    # extract "asin" from Surprise's generated item format
    if mf_based:
        test_recommendations["pred_asin"] = test_recommendations["pred_asin"].apply(
            lambda x: [i[0] for i in x]
        )

    # combined test history and recommendations
    test_merged = pd.merge(
        user_rating_history, test_recommendations, on="reviewerID", how="inner"
    )

    # generating recall@k metrics
    test_merged["recall@n"] = test_merged.apply(
        lambda x: recall_at_n(x.asin, x.pred_asin), axis=1
    )
    test_merged["novelty@n"] = test_merged.apply(
        lambda x: novelty_at_n(item_popularity, x.pred_asin, n=n), axis=1
    )
    average_recall_at_n = test_merged["recall@n"].mean()
    average_novelty_at_n = test_merged["novelty@n"].mean()

    print(
        f"The {model_name} has an average recall@{n}: {average_recall_at_n:.5f}, average novelty@{n}: {average_novelty_at_n:.5f}"
    )

    return test_merged
