# How to install

Install the following requirements:
  - python3
  - keras

# Setting up
## Setup train and test data
TODO
### Setup dummy train and test data
```
python dummy_data.py
```

## Get embeddings
- Download pretrained patent word2vec embeddings `patent_w2v.txt` from https://github.com/zenanz/ChemPatentEmbeddings. Put the file in ./resources.
- Download any pretrained fasttext embeddings, e.g. `wiki-news-300d-1M-subword.vec`, from https://fasttext.cc/docs/en/english-vectors.html. Put the file in ./resources.
- TODO: support custom patent pretrained fasttext embeddings.
- TODO: support ELMo.

## create testing embedding subset for fast embedding loading:
```
python create_embedding_subset.py -input_fname ./resources/patent_w2v.txt -output_fname=./resources/patent_w2v_small.txt -subset_size 20
python create_embedding_subset.py -input_fname ./resources/wiki-news-300d-1M-subword.vec -output_fname=./resources/wiki-news-300d-1M-subword-small.vec -subset_size 20
```
# Train
Note that the commands are using the testing embedding subset for model completeness testing. To use the full embeddings, replace the filename with the path to the full embeddings. 
w2v only:
```
python train.py -w2v_embedding="patent_w2v_small.txt" -w2v_embedding_dim=200 -embedding_dim=200
```

fasttext only:
```
python train.py -fasttext_embedding="wiki-news-300d-1M-subword-small.vec" -fasttext_embedding_dim=300 -embedding_dim=300
```

fasttext+w2v concat at start raw:
```
python train.py -w2v_embedding="patent_w2v_small.txt" -w2v_embedding_dim=200 -fasttext_embedding="wiki-news-300d-1M-subword-small.vec" -fasttext_embedding_dim=300 -concat_at_start 1 -concat_method='raw' -embedding_dim=500
```

# Predict
## Predict with 1 model
```
python predict.py -checkpoint_paths=relative/checkpoint/path
```

## Predict with multiple models
```
python predict.py -checkpoint_paths=relative/checkpoint/path,another/relative/checkpoint/path
```