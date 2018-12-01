import pandas as pd
from terminaltables import AsciiTable
import matplotlib.pyplot as plt
import numpy as np
import html

class Pipeline:

    def __init__(self):
        self.df = pd.read_csv('dataset.csv', engine='python', error_bad_lines=False)

    def run(self):
        self.__wrangling()
        self.__description()
        self.__visualization()
        self.__feature_engineering()

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

        # replace strings in title
        self.df['title'] = self.df.apply(lambda row: html.unescape(row['title']), axis=1)

    def __visualization(self):
        self.df.hist()
        plt.show()

    def __feature_engineering(self):
        self.df = pd.get_dummies(self.df, prefix=['sku'], columns=['sku']) 
        self.df = pd.get_dummies(self.df, prefix=['delivery_zone'], columns=['delivery_zone']) 
        self.df = pd.get_dummies(self.df, prefix=['shop_id'], columns=['shop_id']) 

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