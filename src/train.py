import logging
import pickle
import sys
import warnings

import click
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sqlalchemy import create_engine

from models.train_model import train_algorithm


@click.command()
@click.option(
    "--category", required=True, type=str, help="Category for recommendations."
)
@click.option("--algorithm", required=True, type=str, help="Recommendation algorithms.")
@click.option("--n", required=True, type=int, help="N-number of items recommended.")
@click.option("--epochs", type=int, default=5, help="Number of training epochs.")
@click.option("--lr", type=float, default=0.005, help="Learning rate.")
@click.option("--beta", type=float, default=0.1, help="Regularisation rate.")
@click.option(
    "--input_path",
    type=str,
    default="data/evaluation",
    help="Path to training dataset.",
)
@click.option(
    "--output_path", type=str, default=".", help="Path to save recommendations."
)
@click.option(
    "--d2v_path", type=str, default="models/d2v", help="Path to trained doc2vec model."
)
@click.option(
    "--lda_path", type=str, default="models/lda", help="Path to trained LDA model."
)
def main(
    category: str,
    algorithm: str,
    n: str,
    epochs: int,
    lr: float,
    beta: float,
    input_path: str,
    output_path: str,
    d2v_path: str,
    lda_path: str,
):
    """Train recommender algorithms to generate top-N recommendation list."""

    # reproducibility
    SEED = 42
    np.random.seed(SEED)

    # logging
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(message)s", handlers=[stream_handler]
    )

    logging.info("[1/5]    Loading train dataset...")
    train = pd.read_csv(f"{input_path}/{category}_train.csv")

    # loading necessary saved models
    logging.info("[2/5]    Loading necessary saved models (e.g. Doc2Vec, LDA)...")
    # loading d2v
    item_d2v = Doc2Vec.load(f"{d2v_path}/{category}_item_50_10_d2v.model")
    user_item_d2v = Doc2Vec.load(f"{d2v_path}/{category}_user_item_50_10_d2v.model")
    # loading lda
    sys.path.append(r".")  # prevent "ModuleNotFoundError"
    lda = pickle.load(open(f"{lda_path}/{category}_lda.model", "rb"))

    # instantiate algorithms and training loops
    logging.info(f"[3/5]    Training {algorithm} model...")
    logging.info("Estimated training time: 15-60mins, on 1.4GHz i5, 16GB Ram.")
    TRAINING_PARAMS = {
        "algorithm": algorithm,
        "train": train,
        "n": n,
        "item_d2v": item_d2v,
        "user_item_d2v": user_item_d2v,
        "lda": lda,
        "epochs": epochs,
        "beta": beta,
        "lr": lr,
    }
    top_ns = train_algorithm(**TRAINING_PARAMS)

    # transform top_ns into dataframe
    try:
        top_ns_df = pd.DataFrame.from_dict(top_ns).T.reset_index()
        top_ns_df = top_ns_df.melt(
            id_vars="index", var_name="item_rank", value_name="recommended_items"
        ).sort_values(by=["index", "item_rank"])
    except ValueError:
        top_ns_df = (
            pd.DataFrame.from_dict(top_ns, orient="index")
            .stack()
            .to_frame()
            .reset_index()
        )

    # rename columns
    top_ns_df.columns = ["reviewerID", "item_rank", "asin"]
    # add in the column specifying the algorithm
    top_ns_df["algorithm"] = algorithm

    if algorithm in {"ti-mf", "mod-ecf", "funk-svd"}:
        top_ns_df["asin"] = top_ns_df["asin"].apply(lambda x: x[0])

    # getting item title
    item_info = train[["asin", "title"]].drop_duplicates()
    recommendation_df = top_ns_df.merge(item_info, how="left", on="asin")

    # saving to sqlite db
    logging.info("[5/5]    Saving to `test.db`...")
    engine = create_engine(f"sqlite:///{output_path}/test.db", echo=True)
    recommendation_df.to_sql(f"{category}", con=engine, if_exists="append")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
