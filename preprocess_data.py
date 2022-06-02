import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
import os.path
import io
# from pyfasttext import FastText


def readData(file_name, max_length):
    data_path = './resources/' + file_name
    if not os.path.isfile(data_path):
        print(file_name + " not found")
        exit()
    with open(data_path, 'rb') as pckl:
        text = pickle.load(pckl)
        for o, doc in enumerate(text):
            text[o] = " ".join(text[o].split()[:max_length])
        return text


def editData(text_file, label_file, max_length, tokenizer):
    text_path = './resources/' + text_file
    if not os.path.isfile(text_path):
        print(text_file + " not found")
        exit()
    with open(text_path, 'rb') as pckl:
        texts = pickle.load(pckl)
    for o, doc in enumerate(texts):
        texts[o] = " ".join(texts[o].split()[:max_length])
    sequences = tokenizer.texts_to_sequences(texts)
    del texts
    data = pad_sequences(sequences, maxlen=max_length,
                         padding='post', truncating='post')
    del sequences
    label_path = './resources/' + label_file
    if not os.path.isfile(label_path):
        print(label_file + " not found")
        exit()
    with open(label_path, 'rb') as pckl:
        labels = pickle.load(pckl)
    data = data.astype(np.uint16)
    return data, labels


# def preprocess(fasttext_name, embedding_dim, max_length, max_num_words):
#     print("loading FastText data")
#     fastmodel = FastText('./resources/' + fasttext_name)
#     texts_tokenize = readData('train_texts.pkl', max_length)
#     print("Tokenizing data ..")
#     tokenizer = Tokenizer(num_words=max_num_words)
#     tokenizer.fit_on_texts(texts_tokenize)
#     print("Tokenization and fitting done!")
#     print("Loading data ...")
#     x_train, y_train = editData('train_texts.pkl', 'train_labels.pkl',
#                                 max_length, tokenizer)
#     print("Training data loaded")
#     x_val, y_val = editData('test_texts.pkl', 'test_labels.pkl',
#                             max_length, tokenizer)
#     print("Test Data loaded")
#     word_index = tokenizer.word_index
#     print('Preparing embedding matrix ...')
#     embedding_matrix = np.zeros((max_num_words, embedding_dim))
#     for word, i in word_index.items():
#         if i >= max_num_words:
#             continue
#         embedding_matrix[i] = fastmodel[word]
#     print("Embedding done!")
#     return x_train, y_train, x_val, y_val, embedding_matrix

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def preprocess_fasttext_v2(fasttext_name, embedding_dim, max_num_words, tokenizer):
    print("loading FastText data")
    fastmodel = load_vectors('./resources/' + fasttext_name)
    word_index = tokenizer.word_index
    print('Preparing embedding matrix ...')
    embedding_matrix = np.zeros((max_num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        if word in fastmodel:
            embedding_matrix[i] = list(fastmodel[word])
    print("Embedding done!")
    return embedding_matrix

def preprocess_word2vec(word2vec, embedding_dim, max_num_words, tokenizer):
    print("loading word2vec data")
    fastmodel = load_vectors('./resources/' + word2vec)
    word_index = tokenizer.word_index
    print('Preparing embedding matrix ...')
    embedding_matrix = np.zeros((max_num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        if word in fastmodel:
            embedding_matrix[i] = list(fastmodel[word])
    print("Embedding done!")
    return embedding_matrix

def get_train_val_datasets(max_num_words, max_length):
    texts_tokenize = readData('train_texts.pkl', max_length)
    print("Tokenizing data ..")
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts_tokenize)
    print("Tokenization and fitting done!")
    print("Loading data ...")
    x_train, y_train = editData('train_texts.pkl', 'train_labels.pkl',
                                max_length, tokenizer)
    print("Training data loaded")
    x_val, y_val = editData('test_texts.pkl', 'test_labels.pkl',
                            max_length, tokenizer)
    return x_train, y_train, x_val, y_val, tokenizer

def preprocess_elmo(word2vec, embedding_dim, max_length, max_num_words):
    pass
    # print("loading word2vec data")
    # fastmodel = load_vectors('./resources/' + word2vec)
    # texts_tokenize = readData('train_texts.pkl', max_length)
    # print("Tokenizing data ..")
    # tokenizer = Tokenizer(num_words=max_num_words)
    # tokenizer.fit_on_texts(texts_tokenize)
    # print("Tokenization and fitting done!")
    # print("Loading data ...")
    # x_train, y_train = editData('train_texts.pkl', 'train_labels.pkl',
    #                             max_length, tokenizer)
    # print("Training data loaded")
    # x_val, y_val = editData('test_texts.pkl', 'test_labels.pkl',
    #                         max_length, tokenizer)
    # print("Test Data loaded")
    # word_index = tokenizer.word_index
    # print('Preparing embedding matrix ...')
    # embedding_matrix = np.zeros((max_num_words, embedding_dim))
    # for word, i in word_index.items():
    #     if i >= max_num_words:
    #         continue
    #     embedding_matrix[i] = list(fastmodel[word])
    # print("Embedding done!")
    # return x_train, y_train, x_val, y_val, embedding_matrix