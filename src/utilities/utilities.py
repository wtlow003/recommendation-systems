import numpy as np
import pandas as pd
from tqdm import tqdm


def retrieve_nuniques(df, target_col):
    """Retrieve unique summary count of target column in a dataframe.

    Args:
        df [pd.DataFrame]:
        target_col [str]:
    Returns:
        unique_counts [pd.DataFrame]:
    """

    # check unique values
    print(f"The number of unique records in {target_col}: {df[target_col].nunique()}\n")
    unique_counts = df[target_col].value_counts().sort_values(ascending=False).head(10)
    print(unique_counts)
    print("\n")

    return unique_counts


def summary_statistics(df, cols=["reviewerID", "reviewText", "asin"]):
    print(f"The dataframe consists of {df.shape[0]} rows and {df.shape[1]} columns")

    for col in cols:
        print(f"The number of unique {col}: {df[col].nunique()}")


def reviews_count(df, agg="count"):
    """Retrieving aggregrated review counts based on target group.

    Args:
        df [pd.DataFrame]:
        group [str]:
        agg [str]:
    Returns:
        reviews_by_group [pd.DataFrame]
    """

    reviews_by_prod = df.groupby(["asin"]).agg({"reviewText": agg})
    reviews_by_user = df.groupby(["reviewerID"]).agg({"reviewText": agg})

    # summary statistics
    print("==========")
    print(f"Global average ratings: {df['overall'].mean()}")
    print("For product reviews:")
    print(
        f"Minimum reviews for product: {reviews_by_prod.min()[0]}, Maximum reviews for products: {reviews_by_prod.max()[0]}"
    )
    print(f"Average reviews per products: {reviews_by_prod.mean()[0]}")
    print(
        f"The interquartile range:\n{reviews_by_prod['reviewText'].quantile([.25, 0.5, .75])}\n"
    )
    print("For user reviews:")
    print(
        f"Minimum reviews for users: {reviews_by_user.min()[0]}, Maximum reviews for users: {reviews_by_user.max()[0]}"
    )
    print(f"Average reviews per users: {reviews_by_user.mean()[0]}")
    print(
        f"The interquartile range:\n{reviews_by_user['reviewText'].quantile([.25, 0.5, .75])}\n"
    )


def text_statistics(df: pd.DataFrame, text_col: str) -> None:
    """Review text statistics.

    Args:
        df ([pd.DataFrame]): DataFrame consisting of review texts.
        text_col ([str]): Column name for review text.
    """
    # generating text length on characters and word level
    df["reviewCharLength"] = df[text_col].apply(lambda x: len(x))
    df["reviewWordLength"] = df[text_col].apply(lambda x: len(x.split()))

    # statistics by characters
    min_char_length = df["reviewCharLength"].min()
    max_char_length = df["reviewCharLength"].max()
    avg_char_length = df["reviewCharLength"].mean()
    med_char_length = df["reviewCharLength"].median()

    # statistics by words
    min_word_length = df["reviewWordLength"].min()
    max_word_length = df["reviewWordLength"].max()
    avg_word_length = df["reviewWordLength"].mean()
    med_word_length = df["reviewWordLength"].median()

    # print
    print("Text statistics:\n")
    print(
        f"Minimum `reviewText` length: {min_char_length} characters, {min_word_length} words."
    )
    print(
        f"Maximum `reviewText` length: {max_char_length} characters, {max_word_length} words."
    )
    print(
        f"Mean `reviewText` length: {avg_char_length:.2f} characters, {avg_word_length:.2f} words."
    )
    print(
        f"Median `reviewText` length: {med_char_length} characters, {med_word_length} words."
    )


def token_statistics(df: pd.DataFrame, token_col: str) -> None:
    """Post-tokenization statistics.

    Args:
        df ([pd.DataFrame]): DataFrame consisting of review texts.
        token_col ([str]): Column name of tokenized review text.
    """
    # generating token counts
    df["reviewTokenCount"] = df[token_col].apply(lambda x: len(x))

    # statistics by words
    min_word_length = df["reviewTokenCount"].min()
    max_word_length = df["reviewTokenCount"].max()
    avg_word_length = df["reviewTokenCount"].mean()
    med_word_length = df["reviewTokenCount"].median()

    # print
    print("Token statistics:\n")
    print(f"Minimum `reviewText` tokens: {min_word_length} tokens.")
    print(f"Maximum `reviewText` tokens: {max_word_length} tokens.")
    print(f"Mean `reviewText` tokens: {avg_word_length:.2f} tokens.")
    print(f"Median `reviewText` tokens: {med_word_length} tokens.")


def retrieve_recommendations(
    train: pd.DataFrame, top_ns: dict, target_user: str = None, mf_based: bool = False
):
    """ """
    if not target_user:
        # generating a random user
        target_user = np.random.choice(list(train["reviewerID"].unique()), 1)[0]
    print(f"For user: {target_user}:")
    print(
        f"Purchase History:\n{train[train['reviewerID'] == target_user][['asin', 'title']]}"
    )

    # find the recommendations
    print(f"\nRecommending:\n")
    if mf_based:
        recommendations = (
            train[train["asin"].isin([i[0] for i in top_ns[target_user]])][
                ["asin", "title"]
            ]
            .drop_duplicates(subset="asin")
            .set_index("asin")
        )
        print(
            f"{recommendations.loc[[i[0] for i in top_ns[target_user]]].reset_index()}"
        )
    else:
        recommendations = (
            train[train["asin"].isin(top_ns[target_user])][["asin", "title"]]
            .drop_duplicates(subset="asin")
            .set_index("asin")
        )
        print(f"{recommendations.loc[top_ns[target_user]].reset_index()}")


def get_topic_vectors(model, corpus: list, n_topics: int = 50) -> list:
    """Inferring Topic Modelling vectors given review.

    Args:
        model ([LDA]): Trained Latent Dirchlet Allocation (LDA) topic model.
        corpus ([list]): Review corpus, by either user or item level.
        n_topics ([int]): Fixed number of topics per review. Default is ``50``.
    """
    topic_vecs = []
    for i in range(len(corpus)):
        top_topics = model.get_document_topics(corpus[i])
        topic_vecs.append([top_topics[i][1] for i in range(n_topics)])

    return topic_vecs


def generate_user_item_vectors(train: pd.DataFrame, lda) -> tuple:
    """Generate user and item vectors and relevant mapping to initialize Matrix Factorization.

    Args:
        lda ([LDA]): Trained Latent Dirchlet Allocation (LDA) topic model.
        train ([pd.DataFrame]): Train dataset.
    """
    user_reviews = train.groupby(["reviewerID"])["processedReviewText"].apply(
        lambda x: " ".join(x)
    )
    item_reviews = train.groupby(["asin"])["processedReviewText"].apply(
        lambda x: " ".join(x)
    )

    # get unique users and items
    unique_users = user_reviews.index.tolist()
    unique_items = item_reviews.index.tolist()

    # tokenize reviews
    user_reviews_list = user_reviews.apply(lambda x: x.split()).tolist()
    item_reviews_list = item_reviews.apply(lambda x: x.split()).tolist()

    # generate corpus based on aggregate of user/item reviews
    user_corpus = [lda.dictionary.doc2bow(doc) for doc in user_reviews_list]
    item_corpus = [lda.dictionary.doc2bow(doc) for doc in item_reviews_list]

    # retrieve user and item topics vectors
    user_vecs = get_topic_vectors(lda, user_corpus)
    item_vecs = get_topic_vectors(lda, item_corpus)

    # generate a mapping
    user_idx_map = {k: unique_users[k] for k in range(len(unique_users))}
    item_idx_map = {k: unique_items[k] for k in range(len(unique_items))}
    user_vec_map = {k: v for k, v in zip(unique_users, user_vecs)}
    item_vec_map = {k: v for k, v in zip(unique_items, item_vecs)}

    # loading user topic vectors into DF
    user_vecs = pd.DataFrame.from_dict(user_vec_map, orient="index")
    user_vecs.index.name = "reviewerID"
    # loading item topic vectors into DF
    item_vecs = pd.DataFrame.from_dict(item_vec_map, orient="index")
    item_vecs.index.name = "asin"

    return user_idx_map, user_vecs, item_idx_map, item_vecs


def generate_user_embeddings(user_rating_history: pd.DataFrame, d2v) -> dict:
    """

    Args:
        user_rating_history ([pd.DataFrame]): Train purchase history by user level.
        d2v ([Doc2Vec]): Trained Paragraph Vector model (gensim ``Doc2Vec``).
    """

    # generate unique users
    unique_users = user_rating_history.reset_index()["reviewerID"].tolist()

    user_embeddings = {}
    for user in tqdm(unique_users):
        user_embedding = np.zeros(50)
        for item in user_rating_history[user]:
            user_embedding += d2v.dv[item]

        # computing mean aggregation
        user_embedding /= len(user_rating_history[user])
        user_embeddings[user] = user_embedding

    return user_embeddings


def generate_user_item_embeddings(train: pd.DataFrame, d2v) -> tuple:
    """ """
    # get unique users and items
    unique_users = train["reviewerID"].unique().tolist()
    unique_items = train["asin"].unique().tolist()

    # generating mapping
    user_idx_map = {j: unique_users[j] for j in range(len(unique_users))}
    item_idx_map = {k: unique_items[k] for k in range(len(unique_items))}
    user_vec_map = {j: d2v[j] for j in unique_users}
    item_vec_map = {k: d2v[k] for k in unique_items}

    # loading user d2v vectors into DF
    user_vecs = pd.DataFrame.from_dict(user_vec_map, orient="index")
    user_vecs.index.name = "reviewerID"
    # loading item d2v vectors into DF
    item_vecs = pd.DataFrame.from_dict(item_vec_map, orient="index")
    item_vecs.index.name = "asin"

    return user_idx_map, user_vecs, item_idx_map, item_vecs


def generate_cold_start_users(train: pd.DataFrame):
    """Generate cold starts users based on purchase history of two (2) items."""
    user_purchase_counts = (
        train.groupby("reviewerID").agg({"asin": "count"}).reset_index()
    )
    cold_start_users = user_purchase_counts[user_purchase_counts["asin"] <= 2][
        "reviewerID"
    ].to_list()

    return cold_start_users


def generate_recommendations_df(
    train: pd.DataFrame,
    n_recommendations: dict,
    algo_name: str,
    mf_based=False,
    max_recommended: int = 45,
):
    """ """

    try:
        top_ns = n_recommendations[max_recommended][0]
        top_ns_df = pd.DataFrame.from_dict(top_ns).T.reset_index()
        # transform column data to row wise
        top_ns_df = top_ns_df.melt(
            id_vars="index", var_name="item_rank", value_name="recommended_items"
        ).sort_values(by=["index", "item_rank"])
    except ValueError:
        top_ns = n_recommendations[max_recommended][0]
        top_ns_df = (
            pd.DataFrame.from_dict(
                n_recommendations[max_recommended][0], orient="index"
            )
            .stack()
            .to_frame()
            .reset_index()
        )

    # rename columns
    top_ns_df.columns = ["reviewerID", "item_rank", "asin"]
    # add the in columns specifying top-n and algorithm
    top_ns_df["algorithm"] = algo_name

    if mf_based:
        top_ns_df["asin"] = top_ns_df["asin"].apply(lambda x: x[0])

    # add in item title
    item_info = train[["asin", "title"]].drop_duplicates()
    recommendation_df = top_ns_df.merge(item_info, how="left", on="asin")

    return recommendation_df
