import pandas as pd
from terminaltables import AsciiTable
import matplotlib.pyplot as plt
import numpy as np
import html
import re
import text_normalizer as tn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sn

class Pipeline:

    def __init__(self):
        self.df = pd.read_csv('dataset.csv', engine='python', error_bad_lines=False)

    def run(self):
        self.__wrangling()
        self.__description()
        self.__visualization()
        # self.__feature_engineering()

    def __description(self):
        df_length = len(self.df)

        # table summary
        summary_table_data = [
            ['rows', 'columns'],
            [self.df.shape[0], self.df.shape[1]],
        ]
        summary_table = AsciiTable(summary_table_data)
        print(summary_table.table)

        # columns summary
        columns_table_rows = [['column', 'type', 'missing values']]
        for c in self.df.columns.values:
            columns_table_rows.append(
                [c, self.df.dtypes[c].name, df_length - self.df[c].count()])

        columns_table = AsciiTable(columns_table_rows)

        print(columns_table.table)

        # general summary
        print("General Stats::")
        print(self.df.info())
        print("Summary Stats::" )
        print(self.df.describe())

        # print sample data
        print(self.df.head(10))

    def __wrangling(self):
        # type casting
        self.df['delivery_date'] = pd.to_datetime(self.df.delivery_date, errors="coerce")

        # drop null
        self.df.dropna(inplace=True)

        # delivery date engineering
        self.df['d_year'] = self.df['delivery_date'].apply(lambda d: d.year)
        self.df['d_month'] = self.df['delivery_date'].apply(lambda d: d.month)
        self.df['d_day'] = self.df['delivery_date'].apply(lambda d: d.day)
        self.df['d_day_of_week'] = self.df['delivery_date'].apply(lambda d: d.dayofweek)
        self.df['d_day_name'] = self.df['delivery_date'].apply(lambda d: d.day_name)
        self.df['d_day_of_year'] = self.df['delivery_date'].apply(lambda d: d.dayofyear)
        self.df['d_week_of_year'] = self.df['delivery_date'].apply(lambda d: d.weekofyear)
        self.df['d_quarter'] = self.df['delivery_date'].apply(lambda d: d.quarter)

        # title feature engineering
        self.df['title_word_count'] = self.df['title'].apply(lambda x: len(str(x).split(" ")))
        self.df['title_char_count'] = self.df['title'].str.len()

    def __visualization(self):
        # overall shop and qty sub plot
        fig,ax = plt.subplots() 
        sn.pointplot(data=self.df[['qty', 'd_day_of_week', 'shop_id']],x='d_day_of_week', y='qty', hue='shop_id', ax=ax)
        ax.set(title="Season wise hourly distribution of counts")
        plt.show()

    def __feature_engineering(self):
        # one hot encoding
        self.df = pd.get_dummies(self.df, prefix=['sku'], columns=['sku']) 
        self.df = pd.get_dummies(self.df, prefix=['delivery_zone'], columns=['delivery_zone']) 
        self.df = pd.get_dummies(self.df, prefix=['shop_id'], columns=['shop_id']) 

        # title remove special characters
        self.df['title'] = self.df['title'].apply(lambda x: re.sub(r'\W+', '', x))

        # title lowercase
        self.df['title'] = self.df['title'].apply(lambda x: " ".join(x.lower() for x in x.split()))

        # title unescape html
        self.df['title'] = self.df['title'].apply(lambda x: html.unescape(x))

        # title removing punctuation
        self.df['title'] = self.df['title'].str.replace('[^\w\s]','')

        # title remove stop words
        stop = stopwords.words('english')
        self.df['title'] = self.df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

        # title lemmatization 
        self.df['title'] = self.df['title'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

        # title vectorization
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
        self.df['title_vect'] = list(tfidf.fit_transform(self.df.title).toarray())

    def __feature_scaling(self):
        pass

    def __feature_selection(self):
        pass

    def __modeling(self):
        pass

    def __model_evaluation(self):
        pass

    def __modelTuning(self):
        pass

    def __model_interpretation(self):
        pass

    def __model_deployment(self):
        pass


def cleanup_column_names(df, rename_dict={}, do_inplace=True):
    """This function renames columns of a pandas dataframe
       It converts column names to snake case if rename_dict is not passed.
    Args:
        rename_dict (dict): keys represent old column names and values point to
                            newer ones
        do_inplace (bool): flag to update existing dataframe or return a new one
    Returns:
        pandas dataframe if do_inplace is set to False, None otherwise
    """
    if not rename_dict:
        return df.rename(columns={col: col.lower().replace(' ', '_')
                                  for col in df.columns.values.tolist()},
                         inplace=do_inplace)
    else:
        return df.rename(columns=rename_dict, inplace=do_inplace)