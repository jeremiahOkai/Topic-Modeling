from Labels import Labels
from models import Models
import pandas as pd
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
import scipy
import sqlite3
import json
from time import time

class Preprocess_Main:


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


    def __init__(self, df):
        self.df = df
        # pass

    def create_dict(self):
        mydict = {}
        t0 = time()
        for i in range(len(self.df)):
            currentid = self.df.iloc[i, 0]
            currentvalue = self.df.iloc[i, 1]
            mydict.setdefault(currentid, [])
            mydict[currentid].append(currentvalue)
        print("### Executed time:", round(time() - t0, 3), "s ###")
        print('### Starting to create topics ###')
        return mydict

    @classmethod
    def tokenize(cls, s):
        return ' '.join(cls.tokens_re.findall(s))

    @classmethod
    def replace(cls, s):
        return re.sub(cls.replace_re, '', s)

    @classmethod
    def split(cls, s):
        return s.split()

    @classmethod
    def terms_only(cls, s):
        punctuation = list(string.punctuation)
        stop = stopwords.words('english') + punctuation + ['rt', 'via']
        return [item for item in s if item not in stop and not item.startswith(('#', '@'))]
    # @staticmethod
    def creating_dataframe(self, dictionary):
        final_words = []
        final_words1 = []
        documents = []
        l = []
        z = []
        docs ={}
        keys = dictionary.keys()
        for key in keys:
            kk = str(key)
            k = re.findall(r'\d{8}', kk)
            l.append(k)
        for i in l:
            for j in i:
                z.append(j)
        for key in z:
            # if key == '19234329':
            print("###################### Generating topic labels for {} ############################".format(key))
            df = pd.DataFrame(dictionary[key])
            df.columns = ['Text']
            df_ = df['Text'].apply(lambda x: ''.join(x))
            df_ = df_.str.lower()
            df_ = df_.apply(self.tokenize)
            df_ = df_.apply(self.replace)
            df_ = df_.apply(self.split)
            df_ = df_.apply(self.terms_only)
            df_ = df_.apply(lambda x: ' '.join(x))
            df_ = df_.apply(lambda x: re.sub(r' +', ' ', x))
            [final_words.append("".join(i).strip().split()) for i in df_]
            [final_words1.append(i) for i in final_words if len(i) >= 5]
            [documents.append(re.sub(r' +', " ", (' '.join(i)))) for i in final_words1]

            if key in docs:
                docs[key].append(documents)
            else:
                docs[key] = documents

            mm = Models(5, 10, **docs)
            terms_to_wiki = mm.calling_methods('LDA')
            ll = Labels(terms_to_wiki)
            wiki_titles = ll.get_titles_wiki()
            equal_length = ll.remove_all_null_dicts_returned_from_wiki(**wiki_titles)
            frq = ll.calculating_word_frequency(**equal_length)
            results = ll.predicting_label(**frq)

            print(key, results)
        print('########### FINAL FILE EXECUTED ##################')


        # return docs
