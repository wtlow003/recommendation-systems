import logging
import os
import sys

import click
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from tqdm import tqdm
import yaml


@click.command()
@click.argument("input_filepath", type=str, default="data/evaluation")
@click.argument("output_filepath", type=str, default="models/d2v")
def main(input_filepath, output_filepath: str):

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

    # creating tagged documents
    train_corpus = [
        TaggedDocument(review, [asin])
        for asin, review in list(zip(train["asin"], train["processedReviewText"]))
    ]

    model = Doc2Vec(**MODEL_PARAMS)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # saving model
    os.makedirs(output_filepath, exist_ok=True)
    model.save(f"{output_filepath}/{CATEGORY}_{MODEL_PARAMS['vector_size']}_d2v.model")


if __name__ == "__main__":
    tqdm.pandas()
    main()
