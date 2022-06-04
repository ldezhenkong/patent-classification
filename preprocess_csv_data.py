import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.layers import Dense, Input, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from tensorflow import linalg
from convertXMLtoPCKL import convert_v2
import argparse

categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] 
categories_to_id = {k: v for v, k in enumerate(categories)}

def split_train_and_test(df, train_percent):
    df_filtered = df.dropna(subset=['ipc_single'])
    df_train = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)
    categorizied = df_filtered.groupby(df_filtered['ipc_single'].str[0])
    for _, group in categorizied:
        category_count = len(group)
        train_count = int(train_percent * category_count)
        df_train = pd.concat([df_train, group[:train_count]])
        df_test = pd.concat([df_test, group[train_count:]])
    return df_train.sample(frac=1), df_test.sample(frac=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_test_split', type=float, default=0.9)
    parser.add_argument('-csv', type=str)
    args = parser.parse_args()
    np.random.seed(524)
    df = pd.read_csv(args.csv)
    df_train, df_test = split_train_and_test(df, args.train_test_split)

    convert_v2('train', categories_to_id, df_train)
    convert_v2('test', categories_to_id, df_test)


if __name__ == '__main__':
    main()
