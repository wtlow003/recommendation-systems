import numpy as np
import surprise

from tqdm import tqdm


class EmbeddedMF(surprise.AlgoBase):
    """Latent factors of users and items are generated based off D2V embedding vectors.
       This in turns, allows us to create `P` and `Q` without needing random initialization.

        Args:
            user_map ([dict]): Index-User mapping.
            item_map ([dict]): Index-Item mapping.
            user_factors ([np.array]): Predefined user latent factors initialized using Doc2Vec embeddings.
            item_factors ([np.array]): Predefined item latent factors initialized using Doc2Vec embeddings.
            learning_rate ([float]):
            beta ([float]):
            num_epochs ([int]): Number of training iterations.

        Returns:
            ([None]): Initialized model.
    """

    def __init__(
        self,
        user_map,
        item_map,
        user_factor,
        item_factor,
        learning_rate,
        beta,
        num_epochs,
        num_factors,
    ):
        surprise.AlgoBase.__init__(self)
        self.user_map = {v: k for k, v in user_map.items()}
        self.item_map = {v: k for k, v in item_map.items()}
        self.user_embedding = user_factor
        self.item_embedding = item_factor
        self.alpha = learning_rate
        self.beta = beta
        self.num_epochs = num_epochs
        self.num_factors = num_factors

    def fit(self, train):
        # Instead of random initialization n-latent factors,
        # We initialiazed the latent factors using the D2V aggregated embedding vectors
        # By both user and items, where each embedding is represented by the content
        # of their reviews.
        # This is based on the idea: https://doi.org/10.1145/3383313.3412207
        # Where they initialized the latent factor models using topic vectors generated
        # through NMF.
        surprise.AlgoBase.fit(self, train)
        P = self.user_embedding
        Q = self.item_embedding
        bias_u = np.zeros(len(self.user_embedding))
        bias_i = np.zeros(len(self.item_embedding))
        bias_global = train.global_mean

        for _ in tqdm(range(self.num_epochs)):
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

                    # print(P[ui, :], Q[ii, :], sep='\n')

        self.P = P
        self.Q = Q
        self.bias_u = bias_u
        self.bias_i = bias_i
        self.trainset = train

    def estimate(self, u, i):
        """Returns estimated rating for user u, and item i.

           Prerequisite: Algorithm must be fit to training set.
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

        return est
