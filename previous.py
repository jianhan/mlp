import datetime
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge

from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'


class OrdersPrediction:
    """A simple class will implements the process of orders prediction according to date, delivery zone, order items."""

    def __init__(self, csv):
        """Constructor contains csv path"""
        self.logger = logging.getLogger('ml')
        try:
            self.csv = csv
            self.df = pd.read_csv(csv)
        except Exception as e:
            self.logger.error('error occur while initialize OrdersPrediction' + str(e))

    def __dataCollection(self):
        print("Size: ", self.df.size)
        print("Number of rows::", self.df.shape[0])
        print("Number of columns::", self.df.shape[1])
        print("Column Names::", self.df.columns.values.tolist())
        print("Column Data Types::\n", self.df.dtypes)
        print("Columns with Missing Values::", self.df.columns[self.df.isnull().any()].tolist())
        print("Number of rows with Missing Values::", len(pd.isnull(self.df).any(1).nonzero()[0].tolist()))
        print("Sample Indices with missing data::", pd.isnull(self.df).any(1).nonzero()[0].tolist()[0:5])
        print("General Stats::")
        print(self.df.info())
        print("Summary Stats::")
        print(self.df.describe())
        # self.__cleanup_column_names({'delivery_zone': 'dz'})

    def __dataWrangling(self):
        # trim all
        df_obj = self.df.select_dtypes(['object'])
        self.df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
        self.df[df_obj.columns] = df_obj.apply(lambda x: x.str.lower())

        # filtering data remove all missing values
        self.df['delivery_date'] = pd.to_datetime(self.df.delivery_date, format='%Y-%m-%d', errors='coerce')
        print(self.df.dtypes)
        print("Number of rows::", self.df.shape[0])
        print("Drop Rows with missing dates::")
        self.df = self.df.dropna(subset=['delivery_date'])
        self.df = self.df.dropna(subset=['delivery_zone'])
        self.df = self.df.dropna(subset=['sku'])
        self.df = self.df.dropna(subset=['purchased'])
        print("Columns with Missing Values::", self.df.columns[self.df.isnull().any()].tolist())

        # handling categorical data, delivery zone
        self.df['encoded_delivery_zone'] = self.df.delivery_zone.map(array_to_dict(self.df.delivery_zone.unique()))

        # handling categorical data, skus
        self.df['encoded_sku'] = self.df.sku.map(array_to_dict(self.df.sku.unique()))

        # filter invalid date
        self.df = self.df[self.df['delivery_date'] <= datetime.datetime.now()]

        # drop duplicates
        self.df = self.df.drop_duplicates()

        # generate unique id, following code only required for auto featuring engineering, not applicable at the
        # moment
        # self.df['id'] = np.arange(self.df.shape[0])

        print("Shape--::", self.df.shape)
        print("Sample--::", self.df.head(100))

    def __dataVisualization(self):
        print(self.df.groupby(['sku'])['purchased'].sum())
        print(self.df.groupby(['sku', 'purchased']).agg({'purchased': {'total_purchased': np.sum,
                                                                       'mean_price': np.mean,
                                                                       'variance_price': np.std,
                                                                       'count': np.count_nonzero},
                                                         'purchased': np.sum}))

        # show overall delivery date and purchased status
        self.df.plot(x='delivery_date', y='purchased', style='blue')
        plt.title('Sales status according to delivery date')
        plt.show()

        # show overall delivery zone and purchased status
        self.df[['delivery_zone', 'purchased']].groupby('delivery_zone').sum().plot.barh(color='orange')
        plt.title('Quantities Purchased according to delivery zone')
        plt.show()

        # show histogram diagram for delivery zone and purchased
        self.df[['purchased', 'delivery_zone']].hist(by='delivery_zone', sharex=True)
        plt.show()

        delivery_zone_series = self.df.groupby('delivery_zone').size()
        delivery_zone_series.name = 'Delivery Zone Distribution'
        delivery_zone_series.plot.pie(autopct='%.2f')
        plt.title('Delivery Zone Share')
        plt.show()

    def __featureEngineering(self):
        # Following is a demo of auto featuring, which is not applicable in this case, it is useful when we have
        # multiply tables and join relations, etc... , current data set is too simple and could not get much benifit out of it

        # es = ft.EntitySet(id='sales')
        # es.entity_from_dataframe(entity_id='youfoodz_sales', dataframe=self.df, index='id')
        # feature_matrix, features_defs = ft.dfs(entityset=es, target_entity="youfoodz_sales")
        # print(feature_matrix.head(5))

        # added day month year
        self.df['delivery_year'] = self.df['delivery_date'].map(lambda x: x.year)
        self.df['delivery_month'] = self.df['delivery_date'].map(lambda x: x.month)
        self.df['delivery_day'] = self.df['delivery_date'].map(lambda x: x.day)

        # added total purchased per shop
        self.df['total_purchased_per_shop'] = self.df['purchased'].groupby(self.df['shop_id']).transform('sum')

        # get average purchased according to delivery date
        self.df['purchase_average_by_delivery_date'] = self.df.groupby('delivery_date')['purchased'].transform(
            lambda x: x.mean()
        )

        # get average purchased according to delivery date and sku
        self.df['purchase_average_by_delivery_date_sku'] = self.df.groupby(['delivery_date', 'sku'])[
            'purchased'].transform(
            lambda x: x.mean()
        )

        # get average purchased according to delivery date and sku and shop id
        self.df['purchase_average_by_delivery_date_sku_shop_id'] = self.df.groupby(['delivery_date', 'sku', 'shop_id'])[
            'purchased'].transform(
            lambda x: x.mean()
        )

        # log transformation
        self.df['purchased_og'] = np.log((1 + self.df['purchased']))

    def __modelBuilding(self):
        pass

    def __modelDeployment(self):
        pass

    def __splitDataset(self, df, outcome_labels):
        pass

    def startPipeLine(self):
        """Entry point of start machine learning process"""
        # try:
        self.__dataCollection()
        self.__dataWrangling()
        self.__featureEngineering()
        self.__dataVisualization()
        # print(self.df.columns)
        # split training features and output labels
        outcome_label = self.df['purchased']

        feature_names = ['shop_id',
                         'encoded_delivery_zone', 'encoded_sku', 'delivery_year',
                         'delivery_month', 'delivery_day', 'total_purchased_per_shop',
                         'purchase_average_by_delivery_date',
                         'purchase_average_by_delivery_date_sku',
                         'purchase_average_by_delivery_date_sku_shop_id', 'purchased_og']
        training_features = self.df[feature_names]
        X_train, X_test, y_train, y_test = train_test_split(training_features, outcome_label,
                                                            test_size=0.33,
                                                            random_state=42)
        ss = StandardScaler()
        X_train_std = ss.fit_transform(X_train)
        X_test_std = ss.transform(X_test)

        model = self.__linerRegression(X_train_std, X_test_std, y_train, y_test)

        # self.__lasso(X_train_std, X_test_std, y_train, y_test)
        # self.__bayesian_ridge_regression(X_train_std, X_test_std, y_train, y_test)

        import os
        from sklearn.externals import joblib
        if not os.path.exists('Model'):
            os.mkdir('Model')
        if not os.path.exists('Scaler'):
            os.mkdir('Scaler')
        joblib.dump(model, r'Model/model.pickle')
        joblib.dump(ss, r'Scaler/scaler.pickle')

        # except Exception as e:
        # except Exception as e:
        #     self.logger.error('error occur while running pipeline::' + str(e))

    def __linerRegression(self, X_train, X_test, y_train, y_test):
        lm = LinearRegression()
        model = lm.fit(X_train, y_train)
        print('Liner Regression Accuracy:', lm.score(X_test, y_test))
        return model

    def __lasso(self, X_train, X_test, y_train, y_test):
        lm = Lasso(alpha=0.1)
        lm.fit(X_train, y_train)
        print('Lasso Accuracy:', lm.score(X_test, y_test))

    def __bayesian_ridge_regression(self, X_train, X_test, y_train, y_test):
        lm = BayesianRidge()
        lm.fit(X_train, y_train)
        print('BayesianRidge Accuracy:', lm.score(X_test, y_test))

    def __cleanup_column_names(self, rename_dict={}, do_inplace=True):
        """This function renames columns of a pandas dataframe
            It converts column names to snake case if rename_dict is not passed.
        Args:
            rename_dict (dict): keys represent old column names and values point to newer ones
            do_inplace (bool): flag to update existing dataframe or return a new one
        Returns:
            pandas dataframe if do_inplace is set to False, None otherwise
        """
        if not rename_dict:
            return self.df.rename(
                columns={col: col.lower().replace(' ', '_') for col in self.df.columns.values.tolist()},
                inplace=do_inplace)
        else:
            return self.df.rename(columns=rename_dict, inplace=do_inplace)


def array_to_dict(arr):
    arr.sort()
    retVal = {}
    for i in range(len(arr)):
        retVal[arr[i]] = i
    return retVal


# start ML pipeline
OrdersPrediction("./csvs/test.csv").startPipeLine()
# OrdersPrediction("test.csv").startPipeLine()