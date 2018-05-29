import time
from multiprocessing import Pool
import multiprocessing
import pandas as pd
import re
from nltk.corpus import stopwords
import string
import sqlite3
import eventregistry as ER
import numpy as np
import csv
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import wikipedia as wiki
import nltk
# nltk.download('stopwords')
from nltk.util import ngrams
import numpy as np
import re
import urllib.parse
from collections import Counter
import operator
import requests
import pprint as pp
from itertools import islice
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import ast
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
import codecs
from sklearn import feature_extraction
import sklearn.cluster
import distance


def creating_dbFiles(data):
    chunksize = 10 ** 6

    filename = ''  # put filename here

    path = ''  # put your path here
    conn = sqlite3.connect(path + 'final.db')
    c = conn.cursor()

    print('Creating Database......')
    c.execute('''CREATE TABLE all_users (user_id text, tweets text)''')

    for chunk in pd.read_csv(filename, chunksize=chunksize, engine='python', encoding='utf8', error_bad_lines=False):
        chunk = chunk[['UserID', 'OriginalText']]
        chunk = chunk[np.isfinite(chunk['UserID'])]
        for row in chunk.itertuples():
            c.execute("INSERT INTO all_users VALUES (?,?)", (row[1], row[2]))
    print("Finished with DB")
    conn.commit()
    conn.close()


#     SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = 'YourTableName'

def db_dict(db):
    print('Connecting to DBase')
    conn = sqlite3.connect(db)
    c = conn.cursor()
    sql = "SELECT * FROM twitter"
    t0 = time.time()
    print('Executing command')
    df = pd.read_sql(sql, conn)
    print(df.head())
    print('Finised Creating dataFrame')
    print("Executing time:", round(time.time() - t0, 3), "s")
    conn.close()

    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True, how='any', axis=0)
    t0 = time.time()
    print('Creating dictionary for userIDs')
    s = {str(id_): text.tolist() for id_, text in df.groupby('UserID')['Text']}
    print("Executing time:", round(time.time() - t0, 3), "s")
    return s


regex_str = [
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

regex_pat = [
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'  # URL
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
replace_re = re.compile(r'(' + '|'.join(regex_pat) + ')', re.IGNORECASE)


def tokenize(s):
    return ' '.join(tokens_re.findall(s))


def replace(s):
    return re.sub(replace_re, '', s)


def split(s):
    return s.split()


def terms_only(s):
    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'via']
    return [item for item in s if item not in stop and not item.startswith(('#', '@'))]


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def create_dict(df):
    print('Creating dictionary for all user IDs')
    dictList = []
    mydict = {}
    t0 = time.time()
    for row in df.itertuples():
        if row[1] in mydict:
            mydict[row[1]].append(row[2])
        else:
            mydict[row[1]] = [row[2]]

    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    print('Done creating dictionary for all user IDs')
    return mydict


def creating_dataframe(dictionary):
    print('### Starting to create topics ###')
    jj = []
    l = []
    z = []
    timestamps = []
    docs = {}
    t0 = time.time()
    keys = dictionary.keys()
    counter = 1
    for key in keys:
        print('Cleaning Tweets for User {}'.format(counter))
        documents = []
        final_words = []
        final_words1 = []
        df = pd.DataFrame(dictionary[key])
        df.columns = ['Text']
        df_ = df['Text'].apply(lambda x: ''.join(x))
        df_ = df_.str.lower()
        df_ = df_.apply(tokenize)
        df_ = df_.apply(replace)
        df_ = df_.apply(split)
        df_ = df_.apply(terms_only)
        df_ = df_.apply(lambda x: ' '.join(x))
        df_ = df_.apply(lambda x: re.sub(r' +', ' ', x))
        df_ = df_.apply(lambda x: "".join(x).strip().split())
        df_ = df_.apply(lambda x: x if len(x) >= 5 else np.nan)
        df_.dropna(inplace=True)
        df_ = df_.apply(lambda x: re.sub(r' +', " ", (' '.join(x))))

        if key in docs:
            docs[key].append(df_.values)
        else:
            docs[key] = df_.values
        counter += 1
    jj.append(docs)
    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    print('Done creating dataframe for all users')
    return jj


def LDA_model(results):
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    print('Working on LDA models')
    documents = results
    get_all_docs = {}
    counter = 1
    t0 = time.time()
    for k, v in documents.items():
        print('Creating LDA_model for User', counter, ' to dictionary')
        documents = v
        tf_vectorizer = CountVectorizer(max_df=0.5, min_df=1, stop_words='english')
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()
        lda = LatentDirichletAllocation(n_components=20, max_iter=10, learning_method='batch',
                                        learning_offset=50., random_state=0).fit(tf)
        if k in get_all_docs:
            get_all_docs[k].append((lda, tf_feature_names))
        else:
            get_all_docs[k] = [(lda, tf_feature_names)]
        counter += 1
    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    print('Done creating all LDA models')
    return get_all_docs


def display_topics(model):
    print('Creating topics for LDA models')
    terms_to_wiki = {}
    t0 = time.time()
    counter = 0
    try:
        for k, v in model.items():
            print('Getting topics from LDA for User', counter, ' to dictionary')
            topics_dict = {}
            for topic_idx, topic in enumerate(v[0][0].components_):
                for i in topic.argsort()[:-10 - 1:-1]:
                    if topic_idx in topics_dict:
                        topics_dict[topic_idx].append(v[0][1][i])
                    else:
                        topics_dict[topic_idx] = [v[0][1][i]]
            if k in terms_to_wiki:
                terms_to_wiki[k].append(topics_dict)
            else:
                terms_to_wiki[k] = [topics_dict]
            counter += 1

    except:
        pass
    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    print('Done creating all topics for LDA models')
    return terms_to_wiki


def get_titles_wiki(self):
    print('Getting Wiki tiles')
    titles = {}
    t0 = time.time()
    counter = 1
    for key, value in self.items():
        wiki_titles = {}
        for i in value:
            for k, v in i.items():
                s = ' '.join(v)
                results = wiki.search(s)
                results = results[:1]
                if len(results) != None and len(results) != 0:
                    if k in wiki_titles:
                        wiki_titles[k].append(results)
                    else:
                        wiki_titles[k] = results
        if key in titles:
            titles[key].append(wiki_titles)
        else:
            titles[key] = [wiki_titles]
    #         counter += 1
    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    print('Done creating wiki titles')
    return titles


def return_dict_wiki_topics(**kwargs):
    return kwargs


def return_topic_words_from_model(MyList=[], *args):
    return MyList


def Dmoz(pred):
    final_Dmoz = {}
    t0 = time.time()
    for key, value in pred.items():
        dmozResults = []
        for j in value:
            for k, v in j.items():
                er = ER.EventRegistry(apiKey='32db7607-6c90-40bd-b653-e167da1462c9')
                analytics = ER.Analytics(er)
                cat = analytics.categorize(v[0])
                try:
                    for k, v in cat.items():
                        if k == 'categories':
                            if len(v) != 0 and len(v) != '':
                                for y, value in v[0].items():
                                    if y == 'label':
                                        dmozResults.append(value.split('/')[2])

                except:
                    pass

        with open('/data/s1931628/bigDataFile.csv', 'a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow((key, dmozResults))

    #         if key in final_Dmoz:
    #             final_Dmoz[key].append(dmozResults)
    #         else:
    #             final_Dmoz[key] = dmozResults

    print("### Executed time:", round(time.time() - t0, 3), "s ###")


#     return final_Dmoz

if __name__ == "__main__":
    timestamps = []
    path = '/data/s1931628/'
    s = db_dict(path + 'all_files.db')
    p = Pool(processes=125)
    for items in chunks(s, 100):
        item = {}
        start_time = time.time()
        for key, values in items.items():
            if len(values) < 2000:
                continue
            else:
                if key in item:
                    item[key].append(values[:2000])
                else:
                    item[key] = values[:2000]
        dataFrame = p.map(creating_dataframe, [item])
        for result in dataFrame:
            LDAmodels = p.map(LDA_model, result)
        terms_to_wiki = p.map(display_topics, LDAmodels)
        wiki_titles = p.map(get_titles_wiki, terms_to_wiki)
        dmoz = p.map(Dmoz, wiki_titles)

        with open('/data/s1931628/bigDataFile2.csv', 'a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow((key, dmozResults))
    timestamps.append((time.time() - start_time))
    p.close()
    p.join()

    files_ = []
    with open('/data/s1931628/bigDataFile.csv', 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            x = ast.literal_eval(line[1])
            files_ += x

    countVect = CountVectorizer()
    count_matrix = countVect.fit_transform(files_)  # fit the vectorizer to titles

    # terms is just a list of the features used in the tf-idf matrix.
    terms_count = countVect.get_feature_names()

    # performing clustering
    num_clusters = 4

    km = KMeans(n_clusters=num_clusters)

    km.fit(count_matrix)

    clusters = km.labels_.tolist()

    #     dist is defined as 1 - the cosine similarity of each document.
    #     Cosine similarity is measured against the tf-idf matrix and can be used to generate
    #     a measure of similarity between each document and the other documents in the corpus
    dist = 1 - cosine_similarity(tfidf_matrix)

    #     uncomment the below to save your model
    #     since I've already run my model I am loading from the pickle

    #     joblib.dump(km,  '/data/s1931628/doc_cluster.pkl')
    #     km = joblib.load('/data/s1931628/doc_cluster.pkl')
    #     clusters = km.labels_.tolist()

    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :6]:
            print(' %s' % terms_count[ind])
    print("--- %s seconds ---" % (time.time() - start_time))