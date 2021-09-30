import re

import numpy as np
import pandas as pd

import contractions
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from textblob import TextBlob


def compute_mean_ratings(
    df: pd.DataFrame, count_col: str, rating_col: str
) -> pd.DataFrame:
    """[summary]

    Args:
        df ([type]): [description]
        target_col ([type]): [description]
    Returns:
        [pd.DataFrame]:
    """

    prod_mean_ratings = df.groupby(["asin"]).agg({rating_col: np.mean}).reset_index()
    prod_count_ratings = df.groupby(["asin"]).agg({count_col: "count"}).reset_index()
    df = pd.merge(prod_count_ratings, prod_mean_ratings, how="inner", on="asin")

    df.columns = ["asin", "rating_counts", "rating_average"]

    return df


def compute_weighted_ratings(
    df: pd.DataFrame, count_col: str, avg_rating_col: str, threshold: float = 0.75
) -> pd.DataFrame:
    """Computes weighted ratings based on number of ratings and average rating.

    Args:
        df ([pd.DataFrame]): DataFrame consisting both rating counts and average rating.
        count_col ([str]): Column name for number of ratings.
        avg_rating_col ([str]): Column name for average rating.
        threshold ([float]): Threshold for minimum of ratings/reviews to qualify for ranking.
    Returns:
        [pd.DataFrame]: DataFrame with newly computed weighted ratings
    """

    m = df[count_col].quantile(threshold)
    C = df[avg_rating_col].mean()  # global average rating

    df["rating_weighted_average"] = df.apply(
        lambda x: (
            (x[count_col] / (x[count_col] + m) * x[avg_rating_col])
            + (m / (m + x[count_col]) * C)
        ),
        axis=1,
    )

    return df


def remove_missing_reviews(df: pd.DataFrame, review: str) -> pd.DataFrame:
    """[summary]

    Args:
        df ([type]): [description]
        review ([type]): [description]

    Returns:
        [type]: [description]
    """
    df[review] = df[review].replace({"": np.nan})
    df.dropna(subset=[review], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def lemmatize_with_postags(sentence: str) -> str:
    """Lemmatize a given sentence based on given POS tags.
        Ref: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#comparingnltktextblobspacypatternandstanfordcorenlp

    Args:
        sentence ([type]): [description]

    Returns:
        [type]: [description]
    """
    sent = TextBlob(sentence)
    tag_dict = {"J": "a", "N": "n", "V": "v", "R": "r"}
    words_and_tags = [(w, tag_dict.get(pos[0], "n")) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]

    return " ".join(lemmatized_list)


def preprocess_text(review: str) -> str:
    """Conduct pre-processing for review text.

    The review text will undergo the following pre-processing steps:
        1. Expand contractions
        2. Removing special characters
        3. Lower case
        4. Lemmatization
        5. Exclude stop words
        6. Tokenization

    Args:
        review ([str]): Review text.

    Returns:
        ([str]): Pre-processed review text.
    """
    review = contractions.fix(review)
    review = " ".join(str(review).splitlines())
    review = re.sub(r"[^a-zA-Z]+", " ", review.lower())
    review = review.lower()
    review = lemmatize_with_postags(review)
    review = remove_stopwords(review)
    review = simple_preprocess(review, deacc=True)
    # review = [word for word in review if not word in stopwords.words()]
    # save back into a single string for saving purposes
    review = " ".join(review)

    return review
