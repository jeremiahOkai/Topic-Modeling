from models import Models
from preprocessing import Preprocess_Main
import pandas as pd
from Labels import Labels
import csv
import re
import sys



if __name__ == "__main__":
    df = pd.read_csv('./s.csv')
    df_ = df[['UserID', 'Text']]
    pp = Preprocess_Main(df_)
    user_dictionary = pp.create_dict()
    pp.creating_dataframe(user_dictionary)










