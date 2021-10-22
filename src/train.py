import pickle

import click
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

from models import cf
from utilities import utilities


@click.command()
@click.argument("category", nargs=1)
@click.argument("algorithm", nargs=1)
@click.argument("input_path", type=str, default="data/evaluation")
@click.argument("output_path", type=str, default="./")
@click.argument("d2v_path", type=str, default="../models/d2v")
@click.argument("lda_path", type=str, default="../models/lda")
def main(category, algorithm, input_path, output_path, d2v_path, lda_path):
    """Train recommender algorithms to generate top-N recommendation list."""

    # reproducibility
    SEED = 42
    np.random.seed(42)

    # load training and testing data
    print("[1/4]    Loading train and test dataset...")
    train = pd.read_csv(f"{input_path}/{category}_train.csv")
    test = pd.read_csv(f"{input_path}/{category}_test.csv")

    # loading necessary saved models
    print("[2/4]    Loading necessary saved models (e.g. Doc2Vec, LDA)...")

    # instantiate algorithms and training loops
    print("[3/4]    Training model...")
    if algorithm == "ub-cf":
        model = cf.UserBasedCF()
        model.fit(train, k_neighbours=50)
        candidate_items = model.test()

    print("[4/4]    Generating top-10 recommendations...")
    top_ns = model.get_top_n(n)
    print(top_ns)


if __name__ == "__main__":
    main()
