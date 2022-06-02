from keras.models import load_model
import numpy as np

from preprocess_data import get_train_val_datasets
import argparse
from train import max_length, max_num_words

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint_paths', default="./checkpoint.h5", help="comma separated list of model checkpoints")
    args = parser.parse_args()

    _, _, x_val, _, _ = get_train_val_datasets(max_num_words, max_length)

    # if multiple checkpoint paths, average the predictions
    checkpoint_paths_list = args.checkpoint_paths.split(',')
    model_predictions = []
    for checkpoint_path in checkpoint_paths_list:
        model = load_model(checkpoint_path)

        predictions = model.predict(x_val)
        print(predictions)
        model_predictions.append(predictions)
        # todo: save individual prediction data
    model_predictions = np.array(model_predictions)
    avg_predictions = np.average(model_predictions, axis=0)
    assert avg_predictions.shape == model_predictions[0].shape
    assert np.all(np.sum(avg_predictions, axis=1) == 1.0)
    print(avg_predictions)
    # todo: save aggregated prediction data

if __name__ == '__main__':
    main()

