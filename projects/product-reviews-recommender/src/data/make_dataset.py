import ast
import pandas as pd

from tqdm import tqdm


def calculate_sparsity(df: pd.DataFrame) -> tuple:
    """Calculate the data sparsity based on ratings and reviews.

    Args:
        df ([pd.DataFrame]): DataFrame with counts of `overall` and `reviewText`
            measured against total `reviewerID` * `asin`.

    Returns:
        [tuple]: Tuple of data sparsity wrt. ratings (`overall`) and reviews (`reviewText`).
    """
    # no. of ratings
    rating_numerator = df["overall"].count()
    review_numerator = df["reviewText"].count()

    # number of users and items
    num_users = df["reviewerID"].nunique()
    num_items = df["asin"].nunique()

    denominator = num_users * num_items

    rating_sparsity = (1.0 - (rating_numerator * 1.0) / denominator) * 100
    review_sparsity = (1.0 - (review_numerator * 1.0) / denominator) * 100

    return rating_sparsity, review_sparsity


def _parse_json(json: str) -> dict:
    """Parse json into generator.

    Note: Instead of using json.loads(json), `ast.literal_evcal()` is used
    to prevent errors from occuring when loading metadata json as the json file
    is not saved in json structure defined in python e.g., '' used instead of ""
    for property. Hence, overall performance is slightly slower when loading
    the json files. If metadata is not required, `ujson.loads()` is preferred to
    load the json instead.

    Args:
        json ([str]): Path to json file e.g., '../data/raw/*.json'.

    Returns:
        [generator]: Generator yielding json data.
    """
    with open(json) as f:
        data = f.readlines()
        for line in tqdm(data):
            yield ast.literal_eval(line)


def json_to_df(json: dict) -> pd.DataFrame:
    """Parsing json into DataFrame.

    Args:
        json ([str]): Path to json file e.g., '../data/raw/*.json'.

    Returns:
        [pd.DataFrame]: DataFrame containing parsed json file.
    """
    i = 0
    df = {}
    for review in _parse_json(json):
        df[i] = review
        i += 1

    return pd.DataFrame.from_dict(df, orient="index")
