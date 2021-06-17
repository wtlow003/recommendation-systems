import numpy as np
import pandas as pd


def retrieve_nuniques(df, target_col):
    """Retrieve unique summary count of target column in a dataframe.

        Args:
            df [pd.DataFrame]:
            target_col [str]:
        Returns:
            unique_counts [pd.DataFrame]:
    """

    # check unique values
    print(f"The number of unique records in {target_col}: {df[target_col].nunique()}\n")
    unique_counts = df[target_col].value_counts().sort_values(ascending=False).head(10)
    print(unique_counts)
    print("\n")

    return unique_counts

def summary_statistics(df, cols=['reviewerID', 'reviewText', 'asin']):
    print(f"The dataframe consists of {df.shape[0]} rows and {df.shape[1]} columns")

    for col in cols:
        print(f"The number of unique {col}: {df[col].nunique()}")


def reviews_count(df, agg='count'):
    """Retrieving aggregrated review counts based on target group.

        Args:
            df [pd.DataFrame]:
            group [str]:
            agg [str]:
        Returns:
            reviews_by_group [pd.DataFrame]
    """

    reviews_by_prod = df.groupby(['asin']).agg({'reviewText': agg})
    reviews_by_user = df.groupby(['reviewerID']).agg({'reviewText': agg})

    # summary statistics
    print("==========")
    print(f"Global average ratings: {df['overall'].mean()}")
    print("For product reviews:")
    print(f"Minimum reviews for product: {reviews_by_prod.min()[0]}, Maximum reviews for products: {reviews_by_prod.max()[0]}")
    print(f"Average reviews per products: {reviews_by_prod.mean()[0]}")
    print(f"The interquartile range:\n{reviews_by_prod['reviewText'].quantile([.25, 0.5, .75])}\n")
    print("For user reviews:")
    print(f"Minimum reviews for users: {reviews_by_user.min()[0]}, Maximum reviews for users: {reviews_by_user.max()[0]}")
    print(f"Average reviews per users: {reviews_by_user.mean()[0]}")
    print(f"The interquartile range:\n{reviews_by_user['reviewText'].quantile([.25, 0.5, .75])}\n")

