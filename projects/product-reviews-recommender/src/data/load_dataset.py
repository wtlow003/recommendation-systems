import pandas as pd


def load_dataset(category, input_filepath, load_meta=True):
    """Load `category`'s reviews and metadata.

    Args:
        category ([type]): [description]
        input_filepath ([type]): [description]

    Returns:
        [type]: [description]
    """
    review = pd.read_csv(f"{input_filepath}/{category}_5.csv")

    if load_meta:
        meta = pd.read_csv(f"{input_filepath}/meta_{category}.csv")

        return meta, review
    else:
        review = pd.read_csv(f"{input_filepath}/{category}_5.csv")

        return review
