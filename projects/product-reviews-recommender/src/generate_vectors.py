import logging
import os
import sys

import click
import numpy as np
import pandas as pd
import yaml
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm


@click.command()
@click.argument("input_filepath", type=str, default="data/evaluation")
@click.argument("output_filepath", type=str, default="models/d2v")
def main(input_filepath: str, output_filepath: str):

    # set seed for reproducibility
    SEED = 42
    np.random.seed(SEED)

    # logging
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(message)s", handlers=[stream_handler]
    )

    # read params
    params = yaml.safe_load(open("params.yaml"))["generate_vectors"]

    CATEGORY = params["categories"]
    MODEL_PARAMS = params["d2v_params"]

    logging.info("Loading processed data.")
    train = pd.read_csv(f"{input_filepath}/{CATEGORY}_train.csv")

    logging.info("Generating tagged documents.")
    # split the reviews back into tokens
    train["processedReviewText"] = train["processedReviewText"].progress_apply(
        lambda x: x.split()
    )

    # creating tagged documents for item representation D2V
    item_review_corpus = [
        TaggedDocument(review, [asin])
        for asin, review in list(zip(train["asin"], train["processedReviewText"]))
    ]
    # creating tagged documents for user-item representation D2V
    user_review_corpus = [
        TaggedDocument(review, [reviewerID])
        for reviewerID, review in list(
            zip(train["reviewerID"], train["processedReviewText"])
        )
    ]
    user_item_review_corpus = item_review_corpus + user_review_corpus

    # train item only representation in F-vector dimension space
    item_d2v = Doc2Vec(**MODEL_PARAMS)
    item_d2v.build_vocab(item_review_corpus)
    item_d2v.train(
        item_review_corpus, total_examples=item_d2v.corpus_count, epochs=item_d2v.epochs
    )

    # train user and item representation in F-vector dimension space
    user_item_d2v = Doc2Vec(**MODEL_PARAMS)
    user_item_d2v.build_vocab(user_item_review_corpus)
    user_item_d2v.train(
        user_item_review_corpus,
        total_examples=user_item_d2v.corpus_count,
        epochs=user_item_d2v.epochs,
    )

    # saving model
    os.makedirs(output_filepath, exist_ok=True)
    item_d2v.save(
        f"{output_filepath}/{CATEGORY}_item_{MODEL_PARAMS['vector_size']}_{MODEL_PARAMS['epochs']}_d2v.model"
    )
    user_item_d2v.save(
        f"{output_filepath}/{CATEGORY}_user_item_{MODEL_PARAMS['vector_size']}_{MODEL_PARAMS['epochs']}_d2v.model"
    )


if __name__ == "__main__":
    tqdm.pandas()
    main()
