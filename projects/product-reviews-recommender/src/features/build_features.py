import contractions
import re

from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from textblob import TextBlob


def lemmatize_with_postags(sentence):
    """Lemmatize a given sentence based on given POS tags.
        Ref: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#comparingnltktextblobspacypatternandstanfordcorenlp

    Args:
        sentence ([type]): [description]

    Returns:
        [type]: [description]
    """
    sent = TextBlob(sentence)
    tag_dict = {"J": "a", "N": "n", "V": "v", "R": "r"}
    words_and_tags = [(w, tag_dict.get(pos[0], "n")) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]

    return " ".join(lemmatized_list)


def spelling_correction(sentence):
    """[summary]

    Args:
        sentence ([type]): [description]

    Returns:
        [type]: [description]
    """
    sent = TextBlob(sentence)
    sent = sent.correct()

    return sent


def text_preprocess(review):
    """[summary]

    Args:
        review ([type]): [description]

    Returns:
        [type]: [description]
    """
    # review = spelling_correction(review)
    review = " ".join(str(review).splitlines())  # remove whitespace characters
    review = re.sub(r"http\S+", "", str(review))  # remove links
    review = contractions.fix(review)  # expand contractions
    review = re.sub(r"[^\w\s]", " ", str(review))  # remove punctuations
    review = re.sub(r"'", "", str(review))  # remove single quotes
    review = remove_stopwords(review)
    review = lemmatize_with_postags(review)  # lemmatize sentence
    review = simple_preprocess(review, deacc=True)

    return review
