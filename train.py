import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.layers import Dense, Input, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from tensorflow import linalg
from preprocess_data import get_train_val_datasets, preprocess_fasttext_v2, preprocess_word2vec
import argparse

learning_rate = 0.001145
bs = 128
drop = 0.2584
max_length = 1431
max_num_words = 23140
filters = [6]
num_filters = 2426
nclasses = 8 # categories

def make_embedding_layer(args, tokenizer):
    sequence_input = Input(shape=(max_length,), dtype='uint16')

    if args.concat_at_start and (args.fasttext_embedding is None or args.w2v_embedding is None): 
        raise("need both w2v and fasttext params if concatenating")
    
    if args.fasttext_embedding is not None:
        fasttext_embedding_matrix = preprocess_fasttext_v2(args.fasttext_embedding,args.fasttext_embedding_dim, max_num_words, tokenizer)
        fasttext_embedding_layer = Embedding(max_num_words,
                        args.fasttext_embedding_dim,
                        weights=[fasttext_embedding_matrix],
                        input_length=max_length,
                        trainable=True)

        fasttext_embedded_sequences = fasttext_embedding_layer(sequence_input)
        fasttext_reshape = Reshape((max_length, args.fasttext_embedding_dim, 1))(fasttext_embedded_sequences)
        if not args.concat_at_start:
            assert args.embedding_dim == args.fasttext_embedding_dim, "if using fasttext only, embedding_dim == fasttext_embedding_dim"
            return sequence_input, fasttext_reshape
    if args.w2v_embedding is not None:
        w2v_embedding_matrix = preprocess_word2vec(args.w2v_embedding, args.w2v_embedding_dim, max_num_words, tokenizer)
        w2v_embedding_layer = Embedding(max_num_words,
                        args.w2v_embedding_dim,
                        weights=[w2v_embedding_matrix],
                        input_length=max_length,
                        trainable=True)
        w2v_embedded_sequences = w2v_embedding_layer(sequence_input)
        w2v_reshape = Reshape((max_length, args.w2v_embedding_dim, 1))(w2v_embedded_sequences)
        if not args.concat_at_start:
            assert args.embedding_dim == args.w2v_embedding_dim, "if using w2v only, embedding_dim == w2v_embedding_dim"
            return sequence_input, w2v_reshape
    if args.concat_at_start:
        if args.concat_method == 'raw':
            assert args.embedding_dim == args.fasttext_embedding_dim + args.w2v_embedding_dim, "if raw concat, final embedding dim should equal sum of individual embedding dims"
            reshape = Concatenate(axis=2)([w2v_reshape, fasttext_reshape])
            return sequence_input, reshape
        if args.concat_method == 'linear':
            concat = Concatenate(axis=2)([w2v_embedded_sequences, fasttext_embedded_sequences])
            shrink = Dense(units=args.embedding_dim, activation='linear')(concat)
            reshape = Reshape((max_length, args.embedding_dim, 1))(shrink)
            return sequence_input, reshape
        if args.concat_method == 'svd':
            concat = Concatenate(axis=2)([w2v_embedded_sequences, fasttext_embedded_sequences])
            s = linalg.svd(
                concat, compute_uv=False
            )
            reshape = Reshape((max_length, args.embedding_dim, 1))(s)
            return sequence_input, reshape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fasttext_embedding')
    parser.add_argument('-fasttext_embedding_dim', type=int, default=300)
    parser.add_argument('-w2v_embedding')
    parser.add_argument('-w2v_embedding_dim', type=int, default=200)
    parser.add_argument('-embedding_dim', type=int, default=300)
    parser.add_argument('-concat_at_start', type=bool, default=False)
    parser.add_argument('-concat_method', type=str, default='raw', help='raw|linear|svd')
    parser.add_argument('-checkpoint_path', default="./checkpoint.h5")
    args = parser.parse_args()

    x_train, y_train, x_val, y_val, tokenizer = get_train_val_datasets(max_num_words, max_length)
    sequence_input, reshape = make_embedding_layer(args, tokenizer)

    print("Starting Training ...")

    filter_sizes = []
    for i in filters:
        filter_sizes.append(i)

    maxpool_blocks = []
    for filter_size in filter_sizes:
        conv = Conv2D(num_filters, kernel_size=(filter_size, args.embedding_dim),
                    padding='valid', activation='relu',
                    kernel_initializer='he_uniform',
                    bias_initializer='zeros')(reshape)
        maxpool = MaxPool2D(pool_size=(max_length - filter_size + 1, 1),
                            strides=(1, 1), padding='valid')(conv)
        maxpool_blocks.append(maxpool)

    if len(maxpool_blocks) > 1:
        concatenated_tensor = Concatenate(axis=1)(maxpool_blocks)
    else:
        concatenated_tensor = maxpool_blocks[0]

    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=nclasses, activation='softmax')(dropout)

    model = Model(inputs=sequence_input, outputs=output)

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    y_train = to_categorical(np.asarray(y_train),
                            num_classes=nclasses).astype(np.float16)
    y_val = to_categorical(np.asarray(y_val),
                        num_classes=nclasses).astype(np.float16)

    callbacks = [
        ModelCheckpoint(
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            # The saved model name will include the current epoch.
            filepath=args.checkpoint_path+"_{epoch}",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            verbose=1,
        ),
        EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        )
    ]

    history = model.fit(x_train, y_train,
                        batch_size=bs, shuffle=True,
                        epochs=20,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val))
    # model.save(args.checkpoint_path)

if __name__ == '__main__':
    main()

