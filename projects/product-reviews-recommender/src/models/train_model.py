# import click
import logging
import sys

from datetime import datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    handlers=[stream_handler,],
)


# @click.command()
# @click.argument("output_path", type=str, default="models/d2v/")
def d2v(
    texts,
    category,
    vector_size,
    min_count,
    negative,
    epochs,
    sample=1e-05,
    dm=1,
    workers=8,
):
    """[summary]

    Args:
        texts ([type]): [description]
        category ([type]): [description]
        vector_size ([type]): [description]
        min_count ([type]): [description]
        negative ([type]): [description]
        epochs ([type]): [description]
        sample ([type], optional): [description]. Defaults to 1e-05.
        dm (int, optional): [description]. Defaults to 1.
        workers (int, optional): [description]. Defaults to 8.

    Returns:
        [type]: [description]
    """
    # global logging
    logging.info(f"Training {category}-d2v model")
    # creating tagged documents from lists of texts
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

    # building the model
    model = Doc2Vec(
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        sample=sample,
        dm=dm,
        workers=workers,
    )
    model.build_vocab(documents)

    # training the model
    model.train(documents, total_examples=model.corpus_count, epochs=epochs)
    # save the model
    logging.info(f"Saving {category}-d2v model")
    model.save(f"models/d2v/{category}-d2v.model")

    return documents, model

