from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, AlgoBase, Dataset, Reader
from tqdm import tqdm


class RecommenderBase(ABC):
    """ """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        """Fit learning algorithm to train set."""
        pass

    @abstractmethod
    def test(self):
        """Generate candidate items based on test set."""
        pass


class PreInitialisedMF(AlgoBase):
    """Latent factors of users and items are initialized with fix embedding vectors.
    This in turns, allows us to create `P` and `Q` without needing random initialization.

    The fixed vectors can be generated via both Topic Modelling (pLSA, LDA, NMF) or
    embeddings techniques such as Word2Vec and Paragraph Vector Model (Doc2Vec).

    Usage:
        To instantiate the PreInitialisedMF object:
        >>> ti_mf = PreInitialisedMF()

        To fit model to the training data (e.g., Surprise's trainset):
        >>> ti_mf.fit(trainset, verbose=True)

        To generate rating predictions for all unseen user-item interactions:
        >>> testset = trainset.build_anti_testset()
        >>> predictions = ti_mf.test(testset, verbose=False)

    Args:
        user_map ([dict]): Index-User mapping, e.g., {index: User}.
        item_map ([dict]): Index-Item mapping, e.g., {index: Item}.
        user_factors ([np.array]): Predefined vectors used for initialized latent user
            factors in Matrix Factorization.
        item_factors ([np.array]): Predefined vectors used for initializing latent item
            factors in Matrix Factorization.
        learning_rate ([float]): The learning rate for all parameters. Default is ``0.005``.
        beta ([float]): The regularization term for all parameters. Default is ``0.02``.
        num_epochs ([int]): Number of training iterations. Default is ``10``.
    """

    def __init__(
        self,
        user_map: dict,
        item_map: dict,
        user_factor: np.ndarray,
        item_factor: np.ndarray,
        learning_rate: float = 0.005,
        beta: float = 0.02,
        num_epochs: int = 10,
        num_factors: int = 50,
    ):
        AlgoBase.__init__(self)

        self.user_map = {v: k for k, v in user_map.items()}
        self.item_map = {v: k for k, v in item_map.items()}
        self.user_embedding = user_factor
        self.item_embedding = item_factor
        self.alpha = learning_rate
        self.beta = beta
        self.num_epochs = num_epochs
        self.num_factors = num_factors

    def fit(self, train, verbose=False):
        """Instead of random initialization n-latent factors, we initialiazed the
        latent factors using the fixed vectors generated from topic modelling or
        embedding techniques. We represented both user and items, where each
        embedding is represented by the content of their reviews. This is based
        on the idea: https://doi.org/10.1145/3383313.3412207, where they initialized
        the latent factor models using topic vectors generated through NMF.

        Args:
            train ([surprise.Trainset]): Trainset contains all data that constitute
                a training set used for Surprise's classes.
        """

        AlgoBase.fit(self, train)

        P = self.user_embedding
        Q = self.item_embedding
        bias_u = np.zeros(len(self.user_embedding))
        bias_i = np.zeros(len(self.item_embedding))
        bias_global = train.global_mean

        for current_epoch in range(self.num_epochs):
            if verbose:
                print(f"Processing epoch {current_epoch}")
            for u, i, r_ui in train.all_ratings():
                # retrieving raw uid, iid from iid
                raw_uid = train.to_raw_uid(u)
                raw_iid = train.to_raw_iid(i)

                # locating the index of the user/item vector
                ui = self.user_map[raw_uid]
                ii = self.item_map[raw_iid]

                # compute current error
                dot = sum(P[ui, f] * Q[ii, f] for f in range(self.num_factors))
                err = r_ui - (bias_global + bias_u[ui] + bias_i[ii] + dot)

                # update biases
                bias_u[ui] += self.alpha * (err - self.beta * bias_u[ui])
                bias_i[ii] += self.alpha * (err - self.beta * bias_i[ii])

                # update user and iten latent feature matrices
                for f in range(self.num_factors):
                    P_uf = P[ui, f]
                    Q_if = Q[ii, f]
                    P[ui, f] += self.alpha * (err * Q_if - self.beta * P_uf)
                    Q[ii, f] += self.alpha * (err * P_uf - self.beta * Q_if)

        self.P = P
        self.Q = Q
        self.bias_u = bias_u
        self.bias_i = bias_i
        self.trainset = train

    def estimate(self, u, i, clip=True):
        """Returns estimated rating for user u, and item i.

        Prerequisite: Algorithm must be fit to training set.

        Args:
            u ([type]): The (inner) user id.
            i ([type]): The (inner) item id.
            clip (bool, optional): Clip ratings to minimum and maximum of ``trainset``'s rating
                scale. Defaults to ``True``.
        """

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        est = self.trainset.global_mean

        if known_user:
            est += self.bias_u[u]

        if known_item:
            est += self.bias_i[i]

        if known_user and known_item:
            est += np.dot(self.P[u, :], self.Q[i, :])

        if clip:
            min_rating, max_rating = self.trainset.rating_scale
            est = min(est, max_rating)
            est = max(est, min_rating)

        return est


class FunkMF(RecommenderBase):
    """This class `FunkMF` is purely build on top of Nicholas Hug's `Surprise` package, which
    provides various algorithms to implement recommender systems. We leverage on cythonize
    *SVD* algorithm to increase overall efficiency in training and predicting for over,
    63 million rows at the minimum for our experimental setup.

    Usage:
        To instantiate the FunkMF object:
        >>> funk_mf = FunkMF()

        To fit model to the training data (e.g., Surprise's trainset):
        >>> funk_mf.fit(trainset, verbose=True)

        To generate rating predictions for all unseen user-item interactions:
        >>> testset = trainset.build_anti_testset()
        >>> predictions = funk_mf.test(testset, verbose=False)

    Args:
        n_factors: The number of latent user/item factors. Default is ``50``.
        n_epochs: The number of iterations for SGD optimization. Default is ``10``.
        biased: Whether to use biases. Default is ``True``.
        lr_all: The learning rate for all parameters. Default is ``.005``.
        reg_all: The regularization term for L2 regularization. Default is ``.02``.
        verbose: If ``True``, prints the current epochs. Default is ``True``.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 10,
        biased: bool = True,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        verbose: bool = True,
    ):
        self.algo = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            biased=biased,
            lr_all=lr_all,
            reg_all=reg_all,
            verbose=verbose,
        )
        self.data = None
        self.trainset = None

    def fit(self, train: pd.DataFrame):
        """Fit the training data to the famous *SVD* algorithm, as popularized by `Simon Funk`,
        which is also known as the Funk's Matrix Factorization.

        For our project experimentation, we will include biases and regularize the equation with
        ridge regression (L2) to ensure that we do not overfit the model. This also aligns with
        the proposed review-initialised Matrix Factorization in class `EmbeddedModCF`.

        Args:
            train ([pd.Dataframe]): Training dataset.
        """

        # creating reader toDataFramerating scale
        reader = Reader(rating_scale=(1, 5))
        # generate data require for surprise
        data = Dataset.load_from_df(train[["reviewerID", "asin", "overall"]], reader)
        # generate training set
        trainset = data.build_full_trainset()

        # fitting the trainset to the algorithm SVD (also known as Funk MF)
        self.algo.fit(trainset)

        self.data = data
        self.trainset = trainset

    def test(self, testset, verbose=False):
        """Generate candidate items based on previously unrated items.

        Args:
            verbose: If ``True``, print the current rating prediction for user-item pair.
                Default is ``False``.
        """

        return self.algo.test(testset, verbose=verbose)


class UserBasedCF(RecommenderBase):
    """
    Usage:

    Args:

    """

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
        """Generate user-level similarities matrix based on cosine similarities measure."""

        cosine_sim = cosine_similarity(self.utility_matrix)
        np.fill_diagonal(cosine_sim, 0)
        # generate user similarity matrix
        users_sim = pd.DataFrame(cosine_sim, index=self.utility_matrix.index)
        users_sim.columns = self.utility_matrix.index

        return users_sim

    def __get_k_neighbourhood(self, k_neighbours: float):
        """Retrieve top-k similar users based on descending cosine similarities measure.

        Args:
            k_neighbours ([int]): Number of similiar users ranked in descending order.
        """

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
        """Generate rating predictions for items seen by similar users in k-neighbourhood.

        Args:
            user ([str]): The reviewerID e.g., 'ABCXXXX100SS'
        """

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

    def fit(self, train: pd.DataFrame, k_neighbours: float = 50):
        """Fit learning algorithm to the training set and generate
        a k-neighbourhood of similar users for user-based collaborative
        filtering.

        Args:
            trainset ([pd.DataFrame]): Training dataset e.g., rating/review records.
            k_neighbours ([int]): Number of similiar users ranked in descending order.
        """
        # generate user rating history
        self._rating_history = train.groupby(["reviewerID"])["asin"].apply(list)
        self.utility_matrix = self.__get_utility_matrix(train)
        self.sim_matrix = self.__get_similarities_matrix()
        self._k_neighbourhood = self.__get_k_neighbourhood(k_neighbours)

    def test(self):
        """ """
        # retrieve unique users
        unique_users = self._rating_history.reset_index()["reviewerID"].tolist()

        predictions = {}
        for user in tqdm(unique_users):
            predictions[user] = self.__predict_rating(user)

        return predictions


class EmbeddedItemBasedCF(RecommenderBase):
    """

    Usage:

    Args:

    """

    def __init__(self, d2v: Doc2Vec):
        self.d2v = d2v
        self.user_rating_history = None
        self.user_embeddings = None

    def fit(self, train: pd.DataFrame, dimension: int = 50):
        """Fit learning algorithm to the training set and generate
        user embeddings based on aggregation of previously rated items.

        The user embeddings are used to compute similarities between user
        representation and item representation generated by *Paragraph Vector
        Model* (Doc2Vec).

        Args:
            train ([pd.DataFrame]): Training dataset e.g., rating/review records.
            dimensions ([int]): Length of embeddings trained in ``self.d2v`` model.
        """

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

    def test(self, n: int = 200) -> dict:
        """Generate a list of n-number of candidates items.

        This only generates a generic candidate list of items which do not factor
        in existing rated items and also top-N items required for recommendations.

        Args:
            n ([int]): The number of candidate items to be generated.

        Returns:
            ([dict]): Dictionary of user: n-candidate items.
        """

        candidate_items = {}
        for user in tqdm(self.user_embeddings.items()):
            candidate_items[user[0]] = [
                i for i in self.d2v.dv.most_similar([user[1]], topn=n)
            ]

        return candidate_items
