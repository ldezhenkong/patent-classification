from keras.models import load_model
import numpy as np

from preprocess_data import get_train_val_datasets, load_labels
from preprocess_csv_data import categories
import argparse
from train import max_length, max_num_words
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_paths', default="./checkpoint.h5", help="comma separated list of model checkpoints")
    parser.add_argument('-predictions_save_dir')
    parser.add_argument('-predictions_read_dir')
    parser.add_argument('-plt_title')
    args = parser.parse_args()

    y_val = load_labels("test_labels.pkl")
    print(y_val)
    # if multiple checkpoint paths, average the predictions
    checkpoint_paths_list = args.checkpoint_paths.split(',')
    model_predictions = []
    if args.predictions_read_dir is not None:
        for filename in os.listdir(args.predictions_read_dir):
            f = os.path.join(args.predictions_read_dir, filename)
            predictions = np.load(open(f, 'rb'))
            model_predictions.append(predictions)
    else:
        _, _, x_val, _, _ = get_train_val_datasets(max_num_words, max_length)
        if not os.path.exists(args.predictions_save_dir):
            os.makedirs(args.predictions_save_dir)
        for checkpoint_path in checkpoint_paths_list:
            model = load_model(checkpoint_path)

            predictions = model.predict(x_val)
            print(predictions)
            model_predictions.append(predictions)
            np.save(os.path.join(args.predictions_save_dir, os.path.basename(checkpoint_path)), predictions)
            # todo: save individual prediction data
    model_predictions = np.array(model_predictions)
    avg_predictions = np.average(model_predictions, axis=0)
    assert avg_predictions.shape == model_predictions[0].shape
    print(np.sum(avg_predictions, axis=1))
    # assert np.all(np.sum(avg_predictions, axis=1) == 1.0)
    print(avg_predictions)
    if args.predictions_save_dir is not None and len(checkpoint_paths_list) > 1:
        np.save(os.path.join(args.predictions_save_dir, "aggregated"), avg_predictions)
    
    y_pred = avg_predictions.argmax(axis=-1)
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, display_labels=categories, cmap='BuPu')
    if args.plt_title is not None:
        plt.title(args.plt_title)
    plt.show()

if __name__ == '__main__':
    main()

