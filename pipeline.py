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
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import seaborn as sn

COMMON_WORDS = ['x', '20', '2', '10', '4', '3']

class Pipeline:

    def __init__(self):
        self.df = pd.read_csv(
            'dataset.csv', engine='python', error_bad_lines=False)

    def run(self):
        self.__wrangling()
        self.__description()
        self.__visualization()
        self.__feature_engineering()
        self.__feature_scaling()

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
        print("Summary Stats::")
        print(self.df.describe())

        # print sample data
        print(self.df.head(10))
        

    def __wrangling(self):
        # remove all rows that price is <= .50
        self.df = self.df[self.df['price'] > 0.5]

        # type casting
        self.df['delivery_date'] = pd.to_datetime(
            self.df.delivery_date, errors="coerce")

        # drop null
        self.df.dropna(inplace=True)

        # delivery date engineering
        self.df['d_year'] = self.df['delivery_date'].apply(lambda d: d.year)
        self.df['d_month'] = self.df['delivery_date'].apply(lambda d: d.month)
        self.df['d_day'] = self.df['delivery_date'].apply(lambda d: d.day)
        self.df['d_day_of_week'] = self.df['delivery_date'].apply(
            lambda d: d.dayofweek)
        self.df['d_day_name'] = self.df['delivery_date'].apply(
            lambda d: d.day_name)
        self.df['d_day_of_year'] = self.df['delivery_date'].apply(lambda d: d.dayofyear)
        self.df['d_week_of_year'] = self.df['delivery_date'].apply(lambda d: d.weekofyear)
        self.df['d_quarter'] = self.df['delivery_date'].apply(lambda d: d.quarter)

        # title feature engineering
        self.df['title_word_count'] = self.df['title'].apply(
            lambda x: len(str(x).split(" ")))
        self.df['title_char_count'] = self.df['title'].str.len()

    def __visualization(self):
        # overall shop and qty sub plot

        # fig,ax = plt.subplots()
        # sn.pointplot(data=self.df[['qty', 'd_day_of_week', 'shop_id']],x='d_day_of_week', y='qty', hue='shop_id', ax=ax, estimator=sum)
        # ax.set(title="Shop wise qty distribution of counts")

        # fig,ax = plt.subplots()
        # sn.barplot(data=self.df[['d_month', 'qty']],x='d_month', y='qty', estimator=sum)
        # ax.set(title="Month wise qty distribution of counts")

        # corrMatt = self.df[["d_month", "d_day", "d_day_of_week",
        #                     "price", "d_week_of_year", "shop_id", "delivery_date","qty"]].corr()
        # mask = np.array(corrMatt)
        # mask[np.tril_indices_from(mask)] = False
        # sn.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)

        # sn.stripplot(x="d_day_of_week", y="qty", data=self.df, hue="shop_id", jitter=True, size=.7)
        # sn.stripplot(x="d_day", y="qty", data=self.df, hue="shop_id", jitter=True, size=.7)

        # plt.show()
        pass

    def __feature_engineering(self):

        # one hot encoding on sku
        self.df = pd.get_dummies(self.df, prefix=['sku'], columns=['sku'])

        # one hot encoding on delivery_zone
        self.df = pd.get_dummies(
            self.df, prefix=['delivery_zone'], columns=['delivery_zone'])

        # one hot encoding on shop_id
        self.df = pd.get_dummies(
            self.df, prefix=['shop_id'], columns=['shop_id'])

        # title remove special characters
        self.df['title'] = self.df['title'].apply(
            lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))

        # title lowercase
        self.df['title'] = self.df['title'].apply(
            lambda x: " ".join(x.lower() for x in x.split()))

        # title normalization
        # self.df['title'] = self.df['title'].apply(lambda x: tn.normalize_corpus(x))

        # title unescape html
        self.df['title'] = self.df['title'].apply(lambda x: html.unescape(x))

        # title removing punctuation
        # self.df['title'] = self.df['title'].str.replace('[^\w\s]', '')

        # title remove stop words
        stop = stopwords.words('english')
        self.df['title'] = self.df['title'].apply(
            lambda x: " ".join(x for x in x.split() if x not in stop))

        # title lemmatization
        self.df['title'] = self.df['title'].apply(
            lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

        # remove common words manually, since not all common words are useless
        self.df['title'] = self.df['title'].apply(
            lambda x: " ".join(x for x in x.split() if x not in COMMON_WORDS))
        
        # remove 100 rare words
        rare_words = self.frequent_words(100, False)
        self.df['title'] = self.df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))

        # bag of words
        cv = CountVectorizer(min_df=0., max_df=1.)
        cv_matrix = cv.fit_transform(self.df['title'])
        cv_matrix = cv_matrix.toarray()
        vocab = cv.get_feature_names()
        titleDf = pd.DataFrame(cv_matrix, columns=vocab)
        self.df = pd.concat([self.df, titleDf], axis=1, join_axes=[self.df.index])
        self.df.drop('title', axis=1, inplace=True) 

    def frequent_words(self, count=100, commonOrRare=True):
        if (commonOrRare):
            return pd.Series(' '.join(self.df['title']).split()).value_counts()[:count]
        return pd.Series(' '.join(self.df['title']).split()).value_counts()[-count:]

    def __feature_scaling(self):
        print("General Stats::11111")
        print(self.df.info())
        print("Summary Stats::11111")
        print(self.df.describe())

        # numeric_feature_names = ['qty', 'price', 'd_year', 'd_month', 'd_day', 'd_day_of_week']
        # ss = StandardScaler()
        # ss.fit(self.df[numeric_feature_names])
        # self.df[numeric_feature_names] = ss.transform(self.df[numeric_feature_names])
        # print('-------------------- \n',self.df.head(10))

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
