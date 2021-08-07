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


def summary_statistics(df, cols=["reviewerID", "reviewText", "asin"]):
    print(f"The dataframe consists of {df.shape[0]} rows and {df.shape[1]} columns")

    for col in cols:
        print(f"The number of unique {col}: {df[col].nunique()}")


def reviews_count(df, agg="count"):
    """Retrieving aggregrated review counts based on target group.

    Args:
        df [pd.DataFrame]:
        group [str]:
        agg [str]:
    Returns:
        reviews_by_group [pd.DataFrame]
    """

    reviews_by_prod = df.groupby(["asin"]).agg({"reviewText": agg})
    reviews_by_user = df.groupby(["reviewerID"]).agg({"reviewText": agg})

    # summary statistics
    print("==========")
    print(f"Global average ratings: {df['overall'].mean()}")
    print("For product reviews:")
    print(
        f"Minimum reviews for product: {reviews_by_prod.min()[0]}, Maximum reviews for products: {reviews_by_prod.max()[0]}"
    )
    print(f"Average reviews per products: {reviews_by_prod.mean()[0]}")
    print(
        f"The interquartile range:\n{reviews_by_prod['reviewText'].quantile([.25, 0.5, .75])}\n"
    )
    print("For user reviews:")
    print(
        f"Minimum reviews for users: {reviews_by_user.min()[0]}, Maximum reviews for users: {reviews_by_user.max()[0]}"
    )
    print(f"Average reviews per users: {reviews_by_user.mean()[0]}")
    print(
        f"The interquartile range:\n{reviews_by_user['reviewText'].quantile([.25, 0.5, .75])}\n"
    )


def text_statistics(df: pd.DataFrame, text_col: str) -> None:
    """Review text statistics.

    Args:
        df ([pd.DataFrame]): DataFrame consisting of review texts.
        text_col ([str]): Column name for review text.
    """
    # generating text length on characters and word level
    df["reviewCharLength"] = df[text_col].apply(lambda x: len(x))
    df["reviewWordLength"] = df[text_col].apply(lambda x: len(x.split()))

    # statistics by characters
    min_char_length = df["reviewCharLength"].min()
    max_char_length = df["reviewCharLength"].max()
    avg_char_length = df["reviewCharLength"].mean()
    med_char_length = df["reviewCharLength"].median()

    # statistics by words
    min_word_length = df["reviewWordLength"].min()
    max_word_length = df["reviewWordLength"].max()
    avg_word_length = df["reviewWordLength"].mean()
    med_word_length = df["reviewWordLength"].median()

    # print
    print("Text statistics:\n")
    print(
        f"Minimum `reviewText` length: {min_char_length} characters, {min_word_length} words."
    )
    print(
        f"Maximum `reviewText` length: {max_char_length} characters, {max_word_length} words."
    )
    print(
        f"Mean `reviewText` length: {avg_char_length:.2f} characters, {avg_word_length:.2f} words."
    )
    print(
        f"Median `reviewText` length: {med_char_length} characters, {med_word_length} words."
    )


def token_statistics(df: pd.DataFrame, token_col: str) -> None:
    """Post-tokenization statistics.

    Args:
        df ([pd.DataFrame]): DataFrame consisting of review texts.
        token_col ([str]): Column name of tokenized review text.
    """
    # generating token counts
    df["reviewTokenCount"] = df[token_col].apply(lambda x: len(x))

    # statistics by words
    min_word_length = df["reviewTokenCount"].min()
    max_word_length = df["reviewTokenCount"].max()
    avg_word_length = df["reviewTokenCount"].mean()
    med_word_length = df["reviewTokenCount"].median()

    # print
    print("Token statistics:\n")
    print(f"Minimum `reviewText` tokens: {min_word_length} tokens.")
    print(f"Maximum `reviewText` tokens: {max_word_length} tokens.")
    print(f"Mean `reviewText` tokens: {avg_word_length:.2f} tokens.")
    print(f"Median `reviewText` tokens: {med_word_length} tokens.")
