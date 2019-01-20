import pandas as pd
from sklearn.preprocessing import LabelEncoder


def unique_skus():
    df = pd.read_csv('./csvs/dataset.csv',
                     engine='python', error_bad_lines=False)
    df['sku'] = df['sku'].apply(lambda x: str(x).upper())
    unique_skus = df.sku.unique()

    gle = LabelEncoder()
    sku_labels = gle.fit_transform(df['sku'])
    sku_mappings = {index: label for index, label in enumerate(gle.classes_)}
    df['sku_labels'] = sku_labels
    df.drop('sku', axis=1, inplace=True)


unique_skus()
