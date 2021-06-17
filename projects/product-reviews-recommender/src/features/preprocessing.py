import numpy as np
import pandas as pd


def filter_min_length(df, target_col, threshold=100):
    """Filter target column based on minimum threshold in length.

        Args:
            df [pd.DataFrame]:
            target_col [str]:
        Returns:
            filtered_df [pd.DataFrame]:
    """

    df.dropna(axis=0, subset=[target_col], inplace=True)
    return df[df[target_col].map(len) > 100]


def compute_mean_ratings(df, count_col, rating_col):
    """[summary]

    Args:
        df ([type]): [description]
        target_col ([type]): [description]
    """

    prod_mean_ratings = df.groupby(['asin']).agg({rating_col: np.mean}).reset_index()
    prod_count_ratings = df.groupby(['asin']).agg({count_col: 'count'}).reset_index()
    prod_ratings_merged = pd.merge(prod_count_ratings, prod_mean_ratings, how='inner', on='asin')

    prod_ratings_merged.columns = ['asin', 'rating_counts', 'rating_average']

    return prod_ratings_merged


def compute_weighted_ratings(df, count_col, avg_rating_col, threshold=.75):
    """Computes weighted ratings based on number of ratings and average rating.

        Args:
            df [pd.DataFrame]: DataFrame consisting both rating counts and average rating.
            count_col [str]: Column name for number of ratings.
            avg_rating_col [str]: Column name for average rating.
            threshold [float]: Threshold for minimum of ratings/reviews to qualify for ranking.
        Returns:
            df [pd.DataFrame]: DataFrame with newly computed weighted ratings
    """

    m = df[count_col].quantile(threshold)
    C = df[avg_rating_col].mean()    # global average rating

    df['rating_weighted'] = (df.apply(lambda x: ((x[count_col]/(x[count_col] + m)
                                                  * x[avg_rating_col])
                                                 + (m/(m + x[count_col]) * C)), axis=1)
                            )

    return df
