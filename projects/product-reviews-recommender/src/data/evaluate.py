import math
from datetime import datetime

import pandas as pd


def split_train_test(df: pd.DataFrame, split_ratio: float) -> tuple:
    """Generate time-aware train-test split.

    Args:
        df ([pd.DataFrame]): Processed dataset.
        split_ratio ([float]): Train-test split ratio.

    Returns:
        [Tuple]: Train and test test.
    """
    # generate index
    df.reset_index(inplace=True)

    # change `reviewTime` format
    df["reviewTime"] = df["reviewTime"].apply(
        lambda x: "/".join(x.replace(",", "").split(" "))
    )
    df["reviewTime"] = df["reviewTime"].apply(
        lambda x: datetime.strptime(x, "%m/%d/%Y")
    )

    # sorted by `reviewTime`
    reviews_by_users = (
        df.groupby(["reviewerID"])
        .progress_apply(lambda x: x.sort_values(["reviewTime"], ascending=True))
        .reset_index(drop=True)
    )

    # user by [reviews]
    user_review_map = (
        reviews_by_users.groupby(["reviewerID"])["index"].progress_apply(list).to_dict()
    )

    # generating test records index
    test_review_idx = set()
    for user in user_review_map:
        review_pos_cutoff = math.floor(len(user_review_map[user]) * split_ratio)
        test_review_idx.update(user_review_map[user][review_pos_cutoff:])

    # train/test split
    train = df[~df["index"].isin(test_review_idx)]
    test = df[df["index"].isin(test_review_idx)]

    # removing any user/items not in `train` set
    train_prods = train["asin"].unique()
    train_users = train["reviewerID"].unique()

    test = test[(test["asin"].isin(train_prods) & test["reviewerID"].isin(train_users))]

    return train, test
