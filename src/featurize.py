from datetime import datetime
import logging
import os
from pathlib import Path
import sys

import click
import pandas as pd
import yaml

from features.build_features import preprocess_text, remove_missing_reviews
from pandarallel import pandarallel
from tqdm import tqdm


def _set_logger(log_path):
    """Setting logger for logging code execution.

    Args:
        log_path ([str]): eg: "../log/train.log"

    Returns:
        [logger]: Logging set..
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="a")
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
@click.argument("input_filepath", type=str, default="data/interim")
@click.argument("output_filepath", type=str, default="data/processed")
def main(input_filepath, output_filepath):
    """ """
    # logging
    logger = _set_logger(
        f"log/build_features/build-features-{datetime.today().strftime('%b-%d-%Y')}.log"
    )

    # read params
    params = yaml.safe_load(open("params.yaml"))["featurize"]

    CATEGORY = params["categories"]

    logger.info("Loading merged interim data.")
    products = pd.read_csv(f"{input_filepath}/{CATEGORY}_merged.csv")

    # conduct text pre-processing
    # use pandarallel to parallized func
    products["processedReviewText"] = products["reviewText"].parallel_apply(
        lambda x: preprocess_text(x)
    )
    products = remove_missing_reviews(products, "processedReviewText")

    # dataframe overview
    logger.info(f"{products.info(memory_usage='deep')}")
    logger.info(f"Rows: {products.shape[0]}, Columns: {products.shape[1]}.")

    os.makedirs(output_filepath, exist_ok=True)
    products.to_csv(f"{output_filepath}/{CATEGORY}_processed.csv", index=False)


if __name__ == "__main__":
    pandarallel.initialize()
    # tqdm.pandas()
    main()
