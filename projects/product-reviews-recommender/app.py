import pandas as pd
import streamlit as st
from sqlalchemy import create_engine


def main():
    """ """
    DATA_PATH = "data/evaluation"

    st.title(
        "Leveraging Unsupervised Representation Learning with Reviews for Top-N Recommendations in E-commerce"
    )

    conn = create_engine("sqlite:///recommender.db", echo=False)

    # select category
    st.subheader("Select Category for Recommendations:")
    category_option = st.selectbox("", ("Grocery and Gourmet Food", "Pet Supplies"))

    # load data
    CATEGORY = "_".join(category_option.split(" "))
    DATA = pd.read_csv(f"{DATA_PATH}/{CATEGORY}_train.csv")
    USERS = tuple(DATA["reviewerID"].to_list())

    st.subheader("Select User for Recommendations:")
    user_option = st.selectbox("", USERS)

    # select algorithm for recommendations
    st.subheader("Select Algorithm for Recommendations:")
    algo_option = st.selectbox(
        "",
        ("UB-CF", "RANDOM", "ER-CBF", "FUNK-SVD", "MOD-ECF", "TI-MF"),
    )

    # top-n items for recommendations
    st.subheader("Select N-items Recommended:")
    n_option = st.selectbox("", (10, 25, 30, 45))

    st.write(" ")

    with st.container():
        st.header(f"Generating Recommendations For `{user_option}`:")
        st.subheader("Past Purchase History")
        with st.spinner("Identifying past items"):
            purchase_history = DATA[DATA["reviewerID"] == user_option]
            st.table(purchase_history[["asin", "title"]])
            st.subheader(f"Top-{n_option} Recommended Items:")
            n_recommended = pd.read_sql(
                f"""SELECT *
                    FROM {CATEGORY}
                    WHERE reviewerID = '{user_option}'
                        AND algorithm = '{algo_option}'
                    ORDER BY item_rank
                    LIMIT {n_option}
                """,
                con=conn,
            )
            st.table(n_recommended[["asin", "title"]])
        st.success("Done!")


if __name__ == "__main__":
    main()
