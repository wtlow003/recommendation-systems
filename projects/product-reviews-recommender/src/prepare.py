from datetime import datetime
import logging
import os
from pathlib import Path
import sys

import click
import numpy as np
import pandas as pd
import yaml

from data.make_dataset import calculate_sparsity, json_to_df

def _set_logger(log_path):
    """Setting logger for logging code execution.

    Argss:
        log_path ([str]): eg: "../log/train.log"

    Returns:
        [logger]: Logging set..
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

    logger.info("Finished logger configuration!")

    return logger

@click.command()
@click.argument("input_filepath", type=str, default="data/raw")
@click.argument("output_filepath", type=str, default="data/interim")
def main(input_filepath, output_filepath):
     # logging
    logger = _set_logger(
        f"log/make_dataset/make-dataset-{datetime.today().strftime('%b-%d-%Y')}.log"
    )

    # read params
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    CATEGORY = params["categories"]

    logger.info("Loading raw reviews and metadata.")
    review_json = f"{input_filepath}/{CATEGORY}_5.json"
    metadata_json = f"{input_filepath}/meta_{CATEGORY}.json"

    reviews = json_to_df(review_json)
    metadata = json_to_df(metadata_json)

    # selecting relevant columns only
    rel_review_cols = ["reviewerID", "asin", "overall", "reviewText"]
    rel_metadata_cols = ["asin", "title", "categories"]
    reviews = reviews[rel_review_cols]
    metadata = metadata[rel_metadata_cols]

    # remove rows with unformatted title (i.e. some 'title' may still contain html style content)
    metadata = metadata[~metadata.title.str.contains("getTime", na=False)]

    # merge reviews and metadata
    products = pd.merge(metadata, reviews, how="inner", on="asin")

    # replace all empty string and drop nan
    products.replace("", np.nan, inplace=True)
    products.dropna(
        subset=["reviewerID", "asin", "overall", "reviewText", "title", "categories"],
        inplace=True,
    )

    # dataframe overview
    logger.info(f"{products.info(memory_usage='deep')}")
    logger.info(
        f"Rows: {products.shape[0]}, Columns: {products.shape[1]}, Sparsity: {calculate_sparsity(products)}"
    )

    os.makedirs(output_filepath, exist_ok=True)
    # save file
    products.to_csv(f"{output_filepath}/{CATEGORY}_merged.csv", index=False)


if __name__ == '__main__':
    main()
