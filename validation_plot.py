import pandas as pd

from preprocess_data import get_train_val_datasets, load_labels
from preprocess_csv_data import categories
import argparse
from train import max_length, max_num_words
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv_paths', default="./elmo_only_10000_text_1500.csv", help="comma separated list of validation csvs")
    parser.add_argument('-models', default="elmo_only", help="comma separated list of model names")
    parser.add_argument('-include_train', type=bool, default=False)
    args = parser.parse_args()

    csv_path_list = args.csv_paths.split(',')
    models_list = args.models.split(',')
    dfs = []
    for csv_path in csv_path_list:
        print(csv_path)
        df = pd.read_csv(csv_path)
        dfs.append(df)
    fig, ax = plt.subplots(1,2, figsize=(20, 8))
    title_prefix = "" if args.include_train else "Validation "
    ax[0].set_title(title_prefix + 'Loss')
    ax[0].set_xlabel('Epochs')
    for i, df in enumerate(dfs):
        if args.include_train:
            ax[0].plot(df.index, df.loss, label="train " + models_list[i])
            ax[0].plot(df.index, df.val_loss, label="valid " + models_list[i])
        else:
            ax[0].plot(df.index, df.val_loss, label=models_list[i])
    ax[0].legend()

    ax[1].set_title(title_prefix + 'Accuracy')
    ax[1].set_xlabel('Epochs')
    for i, df in enumerate(dfs):
        if args.include_train:
            ax[1].plot(df.index, df.accuracy, label="train " + models_list[i])
            ax[1].plot(df.index, df.val_accuracy, label="valid " + models_list[i])
        else:
            ax[1].plot(df.index, df.val_accuracy, label=models_list[i])
    ax[1].legend()
    plt.show()

if __name__ == '__main__':
    main()

