from collections import Counter

from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class UserBasedCF:
    """ """

    def __init__(self):
        self._rating_history = None
        self._mean_ratings = None
        self._k_neighbourhood = None
        self.utility_matrix = None
        self.sim_matrix = None

    def __get_utility_matrix(self, trainset: pd.DataFrame):
        """ """
        self._mean_ratings = trainset.groupby(["reviewerID"], as_index=False)[
            "overall"
        ].mean()
        self._mean_ratings.columns = ["reviewerID", "mean_overall"]

        # creating utility matrix
        train = pd.merge(trainset, self._mean_ratings, on="reviewerID")
        # deviation from user's average rating
        train["dev_overall"] = train["overall"] - train["mean_overall"]
        utility_matrix = train.pivot_table(
            index="reviewerID", columns="asin", values="dev_overall"
        )

        return utility_matrix.fillna(utility_matrix.mean(axis=0))

    def __get_similarities_matrix(self):
        """ """
        cosine_sim = cosine_similarity(self.utility_matrix)
        np.fill_diagonal(cosine_sim, 0)
        # generate user similarity matrix
        users_sim = pd.DataFrame(cosine_sim, index=self.utility_matrix.index)
        users_sim.columns = self.utility_matrix.index

        return users_sim

    def __get_k_neighbourhood(self, k_neighbours: float):
        """ """
        # sim_order = np.argsort(self.sim_matrix.values, axis=1)[:, :k_neighbours]
        neighbours = self.sim_matrix.apply(
            lambda x: pd.Series(
                x.sort_values(ascending=False).iloc[:k_neighbours].index,
                index=["top{}".format(i) for i in range(1, k_neighbours + 1)],
            ),
            axis=1,
        )

        return neighbours

    def __predict_rating(self, user):
        """ """
        # retrieve user rating history
        user_rating_history = self._rating_history[user]

        # list of K-neighbourhood of similar users
        sim_users = (
            self._k_neighbourhood[self._k_neighbourhood.index == user]
            .values.squeeze()
            .tolist()
        )
        # retrieve similar user rating history
        sim_users_rating_history = [
            j for i in self._rating_history[sim_users] for j in i
        ]
        # find items rated by similar users by not by target user
        item_under_consideration = set(sim_users_rating_history) - set(
            user_rating_history
        )

        # retrieve target user mean rating
        user_mean_rating = self._mean_ratings.loc[
            self._mean_ratings["reviewerID"] == user, "mean_overall"
        ].values[0]

        candidate_items = {}
        for item in item_under_consideration:
            # retrieve item norm ratings
            item_norm_ratings = self.utility_matrix.loc[:, item]
            # retrieve norm ratings from similar users
            sim_norm_ratings = item_norm_ratings[
                item_norm_ratings.index.isin(sim_users)
            ]
            # retrieve target user and similar user cosine similarities
            corrs = self.sim_matrix.loc[user, sim_users]

            # combined item norm ratings and user corrs - cosine similarities
            user_corrs = pd.concat([sim_norm_ratings, corrs], axis=1)
            user_corrs.columns = ["dev_overall", "correlation"]
            user_corrs["overall"] = user_corrs.apply(
                lambda x: x["dev_overall"] * x["correlation"], axis=1
            )

            # compute predicted ratings
            numerator = user_corrs["overall"].sum()
            denominator = user_corrs["correlation"].sum()
            predict_rating = user_mean_rating + (numerator / denominator)

            candidate_items[item] = predict_rating

        # retrieve counts of items appearing in similar user rating history
        item_counts = pd.DataFrame.from_dict(
            Counter(sim_users_rating_history), orient="index", columns=["count"]
        )
        candidate_items = pd.DataFrame.from_dict(
            candidate_items, orient="index", columns=["pred_overall"]
        )
        # merge predicted ratings and counts
        candidate_items = candidate_items.merge(
            item_counts, left_index=True, right_index=True
        )

        return candidate_items.sort_values(
            by=["count", "pred_overall"], ascending=False
        ).index.tolist()

    def fit(self, trainset: pd.DataFrame, k_neighbours: float = 50):
        """

        Args:
            trainset ([pd.DataFrame]):
            k_neighbours ([int]):
        """
        # generate user rating history
        self._rating_history = trainset.groupby(["reviewerID"])["asin"].apply(list)
        self.utility_matrix = self.__get_utility_matrix(trainset)
        self.sim_matrix = self.__get_similarities_matrix()
        self._k_neighbourhood = self.__get_k_neighbourhood(k_neighbours)

    def predict(self):
        """ """
        # retrieve unique users
        unique_users = self._rating_history.reset_index()["reviewerID"].tolist()

        predictions = {}
        for user in tqdm(unique_users):
            predictions[user] = self.__predict_rating(user)

        return predictions


class EmbeddedMemCF:
    """ """

    def __init__(self, d2v: Doc2Vec):
        self.d2v = d2v
        self.user_rating_history = None
        self.user_embeddings = None

    def fit(self, train: pd.DataFrame, dimension: int = 50):
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

    def predict(self, n: int = 200) -> dict:
        """Generate a list of n-number of candidates items.

        This only generates a generic candidate list of items which do not factor
        in existing rated items and also top-N items required for recommendations.

        """
        candidate_items = {}
        for user in tqdm(self.user_embeddings.items()):
            candidate_items[user[0]] = [
                i for i in self.d2v.dv.most_similar([user[1]], topn=n)
            ]

        return candidate_items
