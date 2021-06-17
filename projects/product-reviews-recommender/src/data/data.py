import json
import gzip
import numpy as np
import pandas as pd


def read_json(json_path):
    """Read json file into dataframe.

        Args:
            json [str]: Path to json dataset.
        Returns:
            df [pd.DataFrame]: Json file loaded into dataframe.
    """

    with open(json_path, 'rb') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data]

    return pd.DataFrame(data)

def read_csv(csv_path):
    """Read csv file into dataframe.

    Args:
        csv_path ([str]): Path to csv dataset.
    """
    pass


def make_dataset(df):
    """Create subset of dataset based on raw data.

        We will break the Amazon reviews dataset into two subsets after initial filtering
        using the criteria of "Verified" = True and False. When user's review is verified,
        this meant that their reviews are associated with a product that is validated by
        Amazon based on their purchase history on the platform. Unverified reviews are
        reviews that cannot be validated by Amazon hence, may poise various issues such as
        potential product boosting through artificially inflating product ratings/reviews,
        intentional demeaning of competitor's product etc.

        Args:
            df [pd.DataFrame]:
        Returns:
            verified [pd.DataFrame]: Dataframe of product reviews that are verified.
            unverified [pd.DataFrame]: Dataframe of product reviews that are unverified.
    """

    verified = df[df['verified']]
    unverified = df[~df['verified']]

    return verified, unverified