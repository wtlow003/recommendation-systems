Leveraging Unsupervised Representation Learning with Reviews to Improve Top-N Recommendation in E-commerce
==============================
Abstract
------------

Traditional recommender systems in E-commerce leverage algorithms such as collaborative filtering to recommend personalised products to users solely based on product ratings by deriving a user-item rating matrix. This arises in two main issues of data sparsity and cold start, which impacts the overall quality of recommendations given the rapid growth of modern E-commerce platforms in both the number of items and users present. The main contribution of this work is the formulation of two approaches, Embedded Review Content-based Filtering (ER-CBF) and Model-based Embedded Collaborative Filtering (MOD-ECF) that incorporates user-generated content such as reviews as both an independent and additional source of information in attempts to tackle concerns of recommendation quality. The unstructured text is represented as document-level embeddings in a continuous feature space of fixed length, using unsupervised representation learning, the Paragraph Vector model. Subsequently, user and item profiles are generated by aggregating the document embeddings on both user and item levels. The resulting representations are used independently in a content-based filtering approach and combined with product ratings in a matrix factorisation collaborative filtering algorithm.

The proposed methodologies are then compared to traditional recommendation algorithms in both accuracy and novelty. Experiments on two categories of a real-world E-commerce dataset demonstrated that ER-CBF outperformed the other systems in terms of accuracy by using only reviews as the sole information while achieving relative novelty. These results suggest that our approach can tackle common problems such as data sparsity and cold start found in traditional recommendation algorithms, thereby indicating the potential of our approach.

Requirements
------------

Getting Started
------------

To run the experiments setup of this project, ensure that following files are populated in their respective folders:
* `data/evaluation/Grocery_and_Gourmet_Food_train.csv`
* `data/evaluation/Grocery_and_Gourmet_Food_test.csv`
* `data/evaluation/Pet_Supplies_train.csv`
* `data/evaluation/Pet_Supplies_test.csv`
* `models/d2v/Grocery_and_Gourmet_Food_item_50_10_d2v.model`
* `models/d2v/Grocery_and_Gourmet_Food_user_item_50_10_d2v.model`
* `models/d2v/Pet_Supplies_item_50_10_d2v.model`
* `models/d2v/Pet_Supplies_user_item_50_10_d2v.model`
* `models/lda/Grocery_and_Gourmet_Food_lda.model`
* `models/lda/Pet_Supplies_lda.model`

If any of the files is **NOT** present, run the following commands to generate all the necessary data and models:
```python
python3 src/prepare.py
python3 src/featurized.py
python3 src/split_train_test.py
python3 src/generate_vectors.py
```
* For the *lda* model, you can refer to the commented-out section `Preparing Topic Vectors [Train/Load]` in:
    * `03-lwt-updated-experimental-setup-ti-mf-ps` or
    * `09-lwt-updated-exprimentatl-setup-ti-mf-ggf`
* After generating the required *lda* model, ensure that you place the model under the directory `models/lda/{category}_lda.model`.


When all files are verified to be in the respective folders, you may run the following **TWELVE (12)** experimental notebooks, each documenting the recommendation process for a specific algorithm (e.g., `UB-CF`) in a one of two categories (`Pet_Supplies` or `Grocery_and_Gourmet_Food`).

A brief description of the notebooks (`.ipynb`) are detailed as the following:

###### For Pet Supplies:
1. `01-lwt-updated-experiment-setup-ub-cf-ps`: User-based Collaborative Filtering.
2. `02-lwt-updated-experiment-setup-funk-svd-ps`: Funk's Matrix Factorisation.
3. `03-lwt-updated-experiment-setup-ti-mf-ps`: Topic-Initialised Matrix Factorisation.
4. `04-lwt-updated-experiment-setup-random-ps`: Random Normal Rating Prediction.
5. `05-lwt-updated-experiment-setup-er-cbf-ps`: (*Proposed*) Embedded Review Content-based Filtering.
6. `06-lwt-updated-experiment-setup-mod-ecf-ps`: (*Proposed*) Model-based Embedded Collborative Filtering.

###### For Groce
7. `07-lwt-updated-experiment-setup-ub-cf-ggf`: User-based Collaborative Filtering.
8. `08-lwt-updated-experiment-setup-funk-svd-ggf`: Funk's Matrix Factorisation.
9. `09-lwt-updated-experiment-setup-ti-mif-ggf`: Topic-Initialised Matrix Factorisation.
10. `10-lwt-updated-experiment-setup-random-ggf`: Random Normal Rating Prediction.
11. `11-lwt-updated-experiment-setup-er-cbf-ggf`: (*Proposed*) Embedded Review Content-based Filtering.
12. `12-lwt-updated-experiment-setup-mod-ecf-ggf`: (*Proposed*) Model-based Embedded Collaborative Filtering.

To execute each experimental setup, simply select under the menu ribbon: `Kernel` -> `Restart & Run All`.
* Note: If you do not with to overwrite any existing saved top-N recommendation lists in `./recommender.db`, simply comment out the code under the section: `Store in SQLite DB`.

Project Overview
------------
### Dataset
------------

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere ipsum ligula, non euismod justo vulputate et. Vestibulum in pharetra est. Sed faucibus, lorem vitae facilisis facilisis, velit risus rutrum nibh, id auctor turpis est at augue. Sed ac dignissim orci. Mauris in felis aliquam, interdum mauris quis, ornare neque. Donec magna dui, auctor id enim non, dignissim malesuada mi. Etiam et purus vehicula dolor scelerisque molestie. Curabitur finibus urna eget tristique congue.

### Data Understanding and Preparation
------------

#### 1. Data Understanding

![Long-tail](reports/figures/long-tail-annotated.png)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere ipsum ligula, non euismod justo vulputate et. Vestibulum in pharetra est. Sed faucibus, lorem vitae facilisis facilisis, velit risus rutrum nibh, id auctor turpis est at augue. Sed ac dignissim orci. Mauris in felis aliquam, interdum mauris quis, ornare neque. Donec magna dui, auctor id enim non, dignissim malesuada mi. Etiam et purus vehicula dolor scelerisque molestie. Curabitur finibus urna eget tristique congue.


#### 2. Data Preparation

![Text Pre-processing Flowchart](reports/figures/text-preprocessing-flowchart.png)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere ipsum ligula, non euismod justo vulputate et. Vestibulum in pharetra est. Sed faucibus, lorem vitae facilisis facilisis, velit risus rutrum nibh, id auctor turpis est at augue. Sed ac dignissim orci. Mauris in felis aliquam, interdum mauris quis, ornare neque. Donec magna dui, auctor id enim non, dignissim malesuada mi. Etiam et purus vehicula dolor scelerisque molestie. Curabitur finibus urna eget tristique congue.

### Proposed Approaches
------------

![Proposed Modelling Approaches in Recommendation Process](reports/figures/recommendation-framework.png)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere ipsum ligula, non euismod justo vulputate et. Vestibulum in pharetra est. Sed faucibus, lorem vitae facilisis facilisis, velit risus rutrum nibh, id auctor turpis est at augue. Sed ac dignissim orci. Mauris in felis aliquam, interdum mauris quis, ornare neque. Donec magna dui, auctor id enim non, dignissim malesuada mi. Etiam et purus vehicula dolor scelerisque molestie. Curabitur finibus urna eget tristique congue.

### Findings
------------

#### 1. `Recall@N` for Overall Users

![Recall@N for Overall Users](reports/metrics/recall@n.png)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere ipsum ligula, non euismod justo vulputate et. Vestibulum in pharetra est. Sed faucibus, lorem vitae facilisis facilisis, velit risus rutrum nibh, id auctor turpis est at augue. Sed ac dignissim orci. Mauris in felis aliquam, interdum mauris quis, ornare neque. Donec magna dui, auctor id enim non, dignissim malesuada mi. Etiam et purus vehicula dolor scelerisque molestie. Curabitur finibus urna eget tristique congue.

#### 2. `Novelty@N` for Overall Users

![Novelty@N for Overall Users](reports/metrics/novelty@n.png)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere ipsum ligula, non euismod justo vulputate et. Vestibulum in pharetra est. Sed faucibus, lorem vitae facilisis facilisis, velit risus rutrum nibh, id auctor turpis est at augue. Sed ac dignissim orci. Mauris in felis aliquam, interdum mauris quis, ornare neque. Donec magna dui, auctor id enim non, dignissim malesuada mi. Etiam et purus vehicula dolor scelerisque molestie. Curabitur finibus urna eget tristique congue.

#### 3. Recall@N for Cold-start Users

![Recall@N for Cold-start Users](reports/metrics/cold_start_recall@n.png)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere ipsum ligula, non euismod justo vulputate et. Vestibulum in pharetra est. Sed faucibus, lorem vitae facilisis facilisis, velit risus rutrum nibh, id auctor turpis est at augue. Sed ac dignissim orci. Mauris in felis aliquam, interdum mauris quis, ornare neque. Donec magna dui, auctor id enim non, dignissim malesuada mi. Etiam et purus vehicula dolor scelerisque molestie. Curabitur finibus urna eget tristique congue.

### Web Application Demo
------------

![Web Application for Exploring Sample Users' Recommendation List](reports/streamlit.gif)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere ipsum ligula, non euismod justo vulputate et. Vestibulum in pharetra est. Sed faucibus, lorem vitae facilisis facilisis, velit risus rutrum nibh, id auctor turpis est at augue. Sed ac dignissim orci. Mauris in felis aliquam, interdum mauris quis, ornare neque. Donec magna dui, auctor id enim non, dignissim malesuada mi. Etiam et purus vehicula dolor scelerisque molestie. Curabitur finibus urna eget tristique congue.

### Conclusion and Future Work
------------

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere ipsum ligula, non euismod justo vulputate et. Vestibulum in pharetra est. Sed faucibus, lorem vitae facilisis facilisis, velit risus rutrum nibh, id auctor turpis est at augue. Sed ac dignissim orci. Mauris in felis aliquam, interdum mauris quis, ornare neque. Donec magna dui, auctor id enim non, dignissim malesuada mi. Etiam et purus vehicula dolor scelerisque molestie. Curabitur finibus urna eget tristique congue.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed, combining transaction log and item information.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── evaluation     <- The train, test split of the final dataset used to train the models and test the predictions.
    │   └── raw            <- The original, immutable data dump. As file is too big, I have provided the
    │                         link to download the raw data seperately in the `data_instruction.txt`.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── d2v            <- Trained and serialized Paragraph Vector model (Gensim's `Doc2Vec`).
    │   └── lda            <- Trained and serialized Latent Dirchlet Allocaton (Gensim's `ldamodel`).
    │
    │
    ├── notebooks
    │   └── exploratory    <- Jupyter notebooks for initial exploration. Naming convention is a
    │                         number (for ordering), the creator's initials, and a short `-`
    │                         delimited description, e.g. `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── reports
    │   └── figures        <- Generated graphics and figures for illustrating concepts.
    │   └── metrics        <- Generated graphics and figures for reporting metrics.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Functions to load, transform, and split raw/merged data.
    │   │   └── evaluate.py
    │   │   └── load_dataset.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Functions to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Functions to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── algorithms.py
    │   │   └── evaluate_model.py
    │   │   └── lda.py
    │   │
    │   └── utilities      <- Functions for miscellaneous tasks, e.g. aggregating document embeddings etc.
    │       └── utilities.py
    │
    ├── presentation       <- Slides used for oral presentation.
    │
    └── literatures        <- Papers cited in the literature review.


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
