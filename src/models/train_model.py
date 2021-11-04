import logging
import sys

import pandas as pd
from gensim.models.doc2vec import Doc2Vec

sys.path.append("../")
from models.algorithms import (
    EmbeddedReviewCBF,
    FunkMF,
    UserBasedCF,
    PreInitialisedMF,
)
from models.lda import LDA
from utilities import utilities


def train_algorithm(
    algorithm: str,
    train: pd.DataFrame,
    n: int,
    item_d2v: Doc2Vec,
    user_item_d2v: Doc2Vec,
    lda: LDA,
    epochs: int,
    beta: float,
    lr: float,
):
    """Train algorithms."""
    if algorithm == "er-cbf":
        model = EmbeddedReviewCBF(item_d2v)
        model.fit(train)
        logging.info("[3/5]    Generating candidate items...")
        candidate_items = model.test()
        logging.info(f"[4/5]    Generating top-{n} recommendations...")
        top_ns = model.get_top_n(n)
    elif algorithm == "funk-svd":
        model = FunkMF(n_epochs=epochs, lr_all=lr, reg_all=beta)
        model.fit(train)
        testset = model.trainset.build_anti_testset()
        logging.info("[3/5]    Generating candidate items...")
        model.test(testset, verbose=False)
        logging.info(f"[4/5]    Generating top-{n} recommendations...")
        top_ns = model.get_top_n(n)
    elif algorithm == "ub-cf":
        model = UserBasedCF()
        model.fit(train)
        logging.info("[3/5]    Generating candidate items...")
        candidate_items = model.test()
        logging.info(f"[4/5]    Generating top-{n} recommendations...")
        top_ns = model.get_top_n(n)
    elif algorithm == "ti-mf":
        (
            user_idx_map,
            user_vecs,
            item_idx_map,
            item_vecs,
        ) = utilities.generate_user_item_vectors(train, lda)
        user_factors, item_factors = user_vecs.to_numpy(), item_vecs.to_numpy()
        model = PreInitialisedMF(
            user_map=user_idx_map,
            item_map=item_idx_map,
            user_factor=user_factors,
            item_factor=item_factors,
            num_epochs=epochs,
            learning_rate=lr,
            beta=beta,
        )
        model.fit(train, verbose=True)
        testset = model.trainset.build_anti_testset()
        logging.info("[3/5]    Generating candidate items...")
        candidate_items = model.test(testset, verbose=False)
        logging.info(f"[4/5]    Generating top-{n} recommendations...")
        top_ns = model.get_top_n(candidate_items, n)
    elif algorithm == "mod-ecf":
        (
            user_idx_map,
            user_vecs,
            item_idx_map,
            item_vecs,
        ) = utilities.generate_user_item_embeddings(train, user_item_d2v)
        user_factors, item_factors = user_vecs.to_numpy(), item_vecs.to_numpy()
        model = PreInitialisedMF(
            user_map=user_idx_map,
            item_map=item_idx_map,
            user_factor=user_factors,
            item_factor=item_factors,
            num_epochs=epochs,
            learning_rate=lr,
            beta=beta,
        )
        model.fit(train, verbose=True)
        testset = model.trainset.build_anti_testset()
        logging.info("[3/5]    Generating candidate items...")
        candidate_items = model.test(testset, verbose=False)
        logging.info(f"[4/5]    Generating top-{n} recommendations...")
        top_ns = model.get_top_n(candidate_items, n)
    else:
        logging.info("Please enter a valid recommendation algorithm!")

    return top_ns
