import csv
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import urllib.parse
from collections import Counter
import wikipedia as wiki
import pprint as pp
from Labels import Labels


class Models:

    def __init__(self, no_topics, no_top_words,  **kwargs):
        self.documents = kwargs
        self.no_topics = no_topics
        self.no_top_words = no_top_words

    def LDA_model(self):
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        for k, v in self.documents.items():
            documents = v
        tf_vectorizer = CountVectorizer(max_df=0.5, min_df=1, stop_words='english')
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()
        lda = LatentDirichletAllocation(n_components=self.no_topics, max_iter=10, learning_method='batch',
                                        learning_offset=50., random_state=0).fit(tf)
        return lda, tf_feature_names, k


    def NMF_model(self):
        # NMF is able to use tf-idf
        for k, v in self.documents.items():
            documents = v
        tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=1, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(documents)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        # Run NMF
        nmf = NMF(n_components=self.no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        return nmf, tfidf_feature_names, k



    def display_topics(self, model, feature_names, key):
        terms_to_wiki = list()
        topics_dict = dict()
        for topic_idx, topic in enumerate(model.components_):
            for i in topic.argsort()[:-self.no_top_words - 1:-1]:
                if topic_idx in topics_dict:
                    topics_dict[topic_idx].append(feature_names[i])
                else:
                    topics_dict[topic_idx] = [feature_names[i]]
        terms_to_wiki.append(topics_dict)

        return terms_to_wiki


    def calling_methods(self, model):
        if model == 'LDA':
            mod, features, key = self.LDA_model()
            terms_to_wiki = self.display_topics(mod,features,key)
            return terms_to_wiki

        elif model == 'NMF':
            mod, features, key = self.NMF_model()
            terms_to_wiki = self.display_topics(mod, features, key)
            return terms_to_wiki

        else:
            print("Such model isnt implemented yet")







