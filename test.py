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
    docs = {}
    t0 = time.time()
    keys = dictionary.keys()
    counter = 1
    for key in keys:
        print('Assigning User', counter, ' to dictionary')
        documents = []
        final_words = []
        final_words1 = []
        print("###################### Generating topic labels for {} ############################".format(key))
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
        [final_words.append("".join(i).strip().split()) for i in df_]
        [final_words1.append(i) for i in final_words if len(i) >= 5]
        [documents.append(re.sub(r' +', " ", (' '.join(i)))) for i in final_words1]

        if key in docs:
            docs[key].append(documents)
        else:
            docs[key] = documents
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
    counter = 0
    t0 = time.time()
    for k, v in documents.items():
        print('Creating LDA_model for User', counter, ' to dictionary')
        documents = v
        tf_vectorizer = CountVectorizer(max_df=0.5, min_df=1, stop_words='english')
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()
        lda = LatentDirichletAllocation(n_topics=20, max_iter=10, learning_method='batch',
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
    #     for j in self:
    for key, value in self.items():
        wiki_titles = {}
        for i in value:
            for k, v in i.items():
                print('Getting wiki titles for User', counter, ' to dictionary')
                s = ' '.join(v)
                results = wiki.search(s)
                results = results[:8]
                if len(results) != None and len(results) != 0:
                    if k in wiki_titles:
                        wiki_titles[k].append(results)
                    else:
                        wiki_titles[k] = results
        if key in titles:
            titles[key].append(wiki_titles)
        else:
            titles[key] = [wiki_titles]
        counter += 1
    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    print('Done creating wiki titles')
    return titles


def return_dict_wiki_topics(**kwargs):
    return kwargs


def return_topic_words_from_model(MyList=[], *args):
    return MyList


def remove_all_null_dicts_returned_from_wiki(terms_to_wiki, **kwargs):
    print('Removing all null values and Getting values in list')
    get_all_values_together = {}
    wiki_titles = kwargs
    terms_to_wiki = terms_to_wiki
    terms_to_wiki = return_topic_words_from_model(terms_to_wiki)
    t0 = time.time()
    counter = 1
    for key, value in wiki_titles.items():
        for kwargs in value:
            for k, v in kwargs.items():
                print('Removing all null values for User', counter, ' to dictionary')
                new_keys_for_topic = {}
                for terms in terms_to_wiki:
                    for words in terms:
                        for q, w in words.items():
                            for next_val in w:
                                for l in next_val.keys():
                                    if int(k) == l:
                                        new_keys_for_topic[k] = (v, next_val[l])
                if key in get_all_values_together:
                    get_all_values_together[key].append(new_keys_for_topic)
                else:
                    get_all_values_together[key] = [new_keys_for_topic]
                counter += 1
    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    print('Done removing all null values and Getting values in list')
    return [get_all_values_together]


def calculating_word_frequency(kwargs):
    print('Calculating frequency for all user ids')
    counter = Counter()
    key_get_frequent_words = dict()
    final_keys = {}
    l = []
    t0 = time.time()
    #     counter = 0
    for key, value in kwargs.items():
        for dict_words in value:
            for k, v in dict_words.items():
                #                 print('Calculating word frequency for User', counter,' to dictionary')
                frequent_words = {}
                for i in v[0]:
                    data = urllib.parse.quote_plus(i)
                    data = re.sub(r'\+', '_', data)
                    URL = "https://en.wikipedia.org/wiki/" + data
                    with urllib.request.urlopen(URL) as source:
                        for line in source:
                            words = re.split(r"[^A-Z]+", line.decode('utf-8'), flags=re.I)
                            counter.update(words)
                        for word in dict_words[k][1]:
                            if i in frequent_words:
                                frequent_words[i].append((word, counter[word]))
                            else:
                                frequent_words[i] = [(word, counter[word])]

                if key in key_get_frequent_words:
                    key_get_frequent_words[key].append(frequent_words)
                else:
                    key_get_frequent_words[key] = [frequent_words]

    for key, value in kwargs.items():
        for dict_words in value:
            for k, v in dict_words.items():
                keys = {}
                for i in v[0]:
                    for q, w in key_get_frequent_words.items():
                        for freq in w:
                            for j in freq.keys():
                                if i == j:
                                    if k in keys:
                                        keys[k].append((j, freq[j]))
                                    else:
                                        keys[k] = [(j, freq[j])]
                if key in final_keys:
                    final_keys[key].append(keys)
                else:
                    final_keys[key] = [keys]
    #                 counter += 1
    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    print('Done calculating frequency for all user ids')
    return final_keys


def predicting_label(kwargs):
    print('Predicting labels for all user ids')
    predicted, final_prediction = {}, {}
    sum = 0
    frq = kwargs
    t0 = time.time()
    #     counter = 0
    for key, value in frq.items():
        for dict_words in value:
            for k, v in dict_words.items():
                #                 print('Predicting labels for User', counter,' to dictionary')
                bigger = dict()
                for i in v:
                    for j in i[1]:
                        sum += j[1]
                    mean = sum / len(v)
                    if k in bigger:
                        bigger[k].append((i[0], mean))
                    else:
                        bigger[k] = [(i[0], mean)]
                if key in predicted:
                    predicted[key].append(bigger)
                else:
                    predicted[key] = [bigger]

    for key, value in predicted.items():
        for predict in value:
            for k, v in predict.items():
                results = []
                counter = 0
                title = ''
                for i in v:
                    if i[1] > counter:
                        counter = i[1]
                        title = i[0]
                results.append((k, title, counter))
                if key in final_prediction:
                    final_prediction[key].append(results)
                else:
                    final_prediction[key] = results
    #                 counter += 1
    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    print('Done predicting labels for all user ids')
    return final_prediction


def DMOZ(results):
    final_Dmoz = {}
    t0 = time.time()
    for key, value in results.items():
        dmozResults = []
        for j in value:
            if type(j) == list:
                for predictions in j:
                    er = ER.EventRegistry(apiKey='32db7607-6c90-40bd-b653-e167da1462c9')
                    analytics = ER.Analytics(er)
                    cat = analytics.categorize(predictions[1])
                    try:
                        for k, v in cat.items():
                            if k == 'categories':
                                if len(v) != 0 and len(v) != '':
                                    for y, value in v[0].items():
                                        if y == 'label':
                                            dmozResults.append(value.split('/')[2])
                    except:
                        pass
            else:
                er = ER.EventRegistry(apiKey='32db7607-6c90-40bd-b653-e167da1462c9')
                analytics = ER.Analytics(er)
                cat = analytics.categorize(j[1])
                try:
                    for k, v in cat.items():
                        if k == 'categories':
                            if len(v) != 0 and len(v) != '':
                                for y, value in v[0].items():
                                    if y == 'label':
                                        dmozResults.append(value.split('/')[2])
                except:
                    pass

        if key in final_Dmoz:
            final_Dmoz[key].append(dmozResults)
        else:
            final_Dmoz[key] = [dmozResults]

    print("### Executed time:", round(time.time() - t0, 3), "s ###")
    return final_Dmoz


print('Connecting to DBase')
conn = sqlite3.connect('/data/s1931628/topicModeling/tweets.db')
c = conn.cursor()
sql = "SELECT * FROM twitter"
t0 = time.time()
print('Executing command')
df = pd.read_sql(sql, conn)
print('Finised Creating dataFrame')
print("Executing time:", round(time.time() - t0, 3), "s")
conn.close()

t0 = time.time()
print('Creating dictionary for userIDs')
s = {str(id_): text.tolist() for id_, text in df.groupby('UserID')['Text']}
print("Executing time:", round(time.time() - t0, 3), "s")

results = []
p = Pool(processes=125)
for item in chunks(s, 500):
    results.append(p.map(creating_dataframe, [item]))
p.close()
p.join()

get_all_docs = []
for result in results:
    for user in result:
        p = Pool(processes=125)
        get_all_docs.append(p.map(LDA_model, user))
    p.close()
    p.join()

terms_to_wiki = []
for i in get_all_docs:
    p = Pool(processes=125)
    terms_to_wiki.append(p.map(display_topics, i))
    p.close()
    p.join()

wiki_titles = []
for i in terms_to_wiki:
    p = Pool(processes = 125)
    wiki_titles.append(p.map(get_titles_wiki,i))
    p.close()
    p.join()

get_all_values_together = []
for i in wiki_titles:
    for j in i:
        get_all_values_together.append(remove_all_null_dicts_returned_from_wiki(terms_to_wiki, **j))

p = Pool(processes=125)
for i in get_all_values_together:
    frq = p.map(calculating_word_frequency, i)
    prediction = p.map(predicting_label, frq)
    dmoz = p.map(DMOZ, prediction)
    p.close()
    p.join()