# -*- coding: utf-8 -*-
import click
import glob
import pandas as pd
import logging
import sys

from data import read_json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


def set_logger(log_path):
    """Setting logger for logging code execution.
    Args:
        log_path [str]: eg: "../log/train.log"
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(f"Finished logger configuration!")

    return logger


def calculate_sparsity(df):
    """[summary]

    Args:
        df ([type]): [description]

    Returns:
        [type]: [description]
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


@click.command()
@click.argument("input_filepath", type=str, default="data/raw")
@click.argument("output_filepath", type=str, default="data/interim/")
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
        transformed data ready for further processing (saved in ../interim).

        This functions transforms raw data from json files into merged csv from
        respective categories' reviews and metadata files.

    Args:
        input_filepath [(str)]: Path to the raw data.
        ouput_filepath ([str]): Path to store the interim data.
    """
    logger = set_logger(
        f"log/make_dataset/make-dataset-{datetime.today().strftime('%b-%d-%Y')}"
    )
    logger.info("Loading raw reviews and metadata into merged data.")

    # loading and unloading paths
    input_filepath, output_filepath = Path(input_filepath), Path(output_filepath)

    # sorting ensures corresponding index between reviews and metadata of categories
    reviews_jsons = sorted([f for f in glob.glob(f"{input_filepath /'*_5.json'}")])
    meta_jsons = sorted([f for f in glob.glob(f"{input_filepath /'meta_*.json'}")])
    cat_names = [str(name).split("/")[-1].split(".")[0] for name in meta_jsons]
    logger.info(f"Reviews jsons: {reviews_jsons}")
    logger.info(f"Meta jsons: {meta_jsons}")

    for i in tqdm(range(len(reviews_jsons))):
        logger.info(f"Transforming: {cat_names[i]}")
        reviews = read_json(f"{reviews_jsons[i]}")
        meta = read_json(f"{meta_jsons[i]}")

        # selecting relevant columns only
        reviews_cols_to_keep = ["overall", "reviewerID", "asin", "reviewText"]
        reviews = reviews[reviews_cols_to_keep]
        logger.info(
            f"Num ratings: {reviews['overall'].count()}, "
            f"reviews: {reviews['reviewText'].count()}, "
            f"users: {reviews['reviewerID'].nunique()}, "
            f"items: {reviews['asin'].nunique()}"
        )

        meta_cols_to_keep = ["title", "brand", "asin"]
        meta = meta[meta_cols_to_keep]
        # ref: https://colab.research.google.com/drive/1Zv6MARGQcrBbLHyjPVVMZVnRWsRnVMpV
        # removing unformatted title (i.e. some 'title' may still contain html style content)
        meta = meta[~meta.title.str.contains("getTime")]

        # logging sparsity
        rating_sparsity, review_sparsity = calculate_sparsity(reviews)
        logger.info(f"Sparsity: {rating_sparsity} rating, {review_sparsity} review.")

        # saving dataframes to output path
        reviews.to_csv(output_filepath / f"{cat_names[i]}_merged.csv", index=False)
        meta.to_csv(output_filepath / f"meta_{cat_names[i]}.csv", index=False)


if __name__ == "__main__":
    main()
