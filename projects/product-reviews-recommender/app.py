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

    # receive `reviewerID` to get recommendations
    st.subheader("Enter Reviewer ID:")
    st.text_input("", key="reviewerID")

    # select category
    st.subheader("Select Category for Recommendations:")
    category_option = st.selectbox("", ("Grocery and Gourmet Food", "Pet Supplies"))

    # load data
    CATEGORY = "_".join(category_option.split(" "))
    DATA = pd.read_csv(f"{DATA_PATH}/{CATEGORY}_train.csv")

    # select algorithm for recommendations
    st.subheader("Select Algorithm for Recommendations:")
    algo_option = st.selectbox(
        "",
        ("UB-CF", "RANDOM", "ER-CBF", "FUNK-SVD", "MOD-ECF"),
    )

    # top-n items for recommendations
    st.subheader("Select N-items Recommended:")
    n_option = st.selectbox("", (10, 25, 30, 45))

    st.write(" ")

    with st.container():
        st.header("Generating Recommendations:")
        st.subheader(f"{st.session_state.reviewerID}'s Past Purchase History")
        with st.spinner("Identifying past items"):
            purchase_history = DATA[DATA["reviewerID"] == st.session_state.reviewerID]
            st.table(purchase_history[["asin", "title"]])
            st.subheader(
                f"{st.session_state.reviewerID} Top-{n_option} Recommended Items:"
            )
            n_recommended = pd.read_sql(
                f"""SELECT *
                    FROM {CATEGORY}
                    WHERE reviewerID = '{st.session_state.reviewerID}'
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
