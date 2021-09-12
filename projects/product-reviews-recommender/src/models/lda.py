import gensim


class LDA:
    def __init__(self, reviews):
        self.reviews = reviews
        self.lda = None
        self.dictionary = None

    def train(self, n_topics=50, n_epochs=20, workers=8):
        # tokenizations
        dictionary = gensim.corpora.Dictionary(self.reviews)
        # filtering tokens less than 5 reviews, more than 0.85 reviews
        dictionary.filter_extremes(no_below=5, no_above=0.85)
        # creating dict how many words and time it appears
        bow_corpus = [dictionary.doc2bow(doc) for doc in self.reviews]

        # train model
        self.lda = gensim.models.LdaMulticore(
            bow_corpus,
            num_topics=n_topics,
            id2word=dictionary,
            passes=n_epochs,
            workers=workers,
        )
        # save dictionary
        self.dictionary = dictionary

    def get_document_topics(self, doc, minimum_probability=0.0):
        """ """
        return self.lda.get_document_topics(
            doc, minimum_probability=minimum_probability
        )
