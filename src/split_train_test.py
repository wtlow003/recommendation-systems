from datetime import datetime
import logging
import os
from pathlib import Path
import sys

import click
import yaml

from data.evaluate import split_train_test
import pandas as pd
from tqdm import tqdm


def _set_logger(log_path: str) -> logging.Logger:
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
@click.argument("input_filepath", type=str, default="data/processed")
@click.argument("output_filepath", type=str, default="data/evaluation")
def main(input_filepath: str, output_filepath: str):

    # logging
    logger = _set_logger(
        f"log/model_selection/dataset-split-{datetime.today().strftime('%b-%d-%Y')}.log"
    )

    # read params
    params = yaml.safe_load(open("params.yaml"))["split_train_test"]

    CATEGORY = params["categories"]
    SPLIT_RATIO = params["split_ratio"]

    logger.info("Loading processed data.")
    products = pd.read_csv(f"{input_filepath}/{CATEGORY}_processed.csv")

    # generate train/test split
    train, test = split_train_test(products, SPLIT_RATIO)

    # dataframe overview
    logger.info(f"{train.info(memory_usage='deep')}")
    logger.info(f"{test.info(memory_usage='deep')}")
    logger.info(f"Trainset: {train.shape}")
    logger.info(f"Testset: {test.shape}")

    os.makedirs(output_filepath, exist_ok=True)
    train.to_csv(f"{output_filepath}/{CATEGORY}_train.csv", index=False)
    test.to_csv(f"{output_filepath}/{CATEGORY}_test.csv", index=False)


if __name__ == "__main__":
    tqdm.pandas()
    main()
