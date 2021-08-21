from gensim.models.doc2vec import Doc2Vec
import numpy as np
from tqdm import tqdm


class EmbeddedCF:
    """ """

    def __init__(self, d2v):
        self.d2v = d2v
        self.user_rating_history = None
        self.user_embeddings = None

    def fit(self, train, dimension=50):
        # get user rating history
        user_rating_history = train.groupby(["reviewerID"])["asin"].apply(list)
        # getting unique users
        unique_users = user_rating_history.reset_index()["reviewerID"].tolist()

        # generating user embeddings for all unique users
        user_embeddings = {}

        for user in tqdm(unique_users):
            user_embedding = np.zeros(dimension)
            for item in user_rating_history[user]:
                user_embedding += self.d2v.dv[item]

            # mean aggregation
            user_embedding /= len(user_rating_history[user])
            user_embeddings[user] = user_embedding

        self.user_rating_history = user_rating_history
        self.user_embeddings = user_embeddings

    def predict(self, n=200):
        """Generate a list of n-number of candidates items.

        This only generates a generic candidate list of items which do not factor
        in existing rated items and also top-N items required for recommendations.

        """
        candidate_items = {}
        for user in tqdm(self.user_embeddings.items()):
            candidate_items[user[0]] = [
                i for i in self.d2v.dv.most_similar([user[1]], topn=n)
            ]

        return candidate_itemse
