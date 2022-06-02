import pickle
from convertXMLtoPCKL import save_data

# create dummy train pickle
train_texts = [
    "this is a patent claim about some chemistry stuff",
    "this is another patent claim about some chemistry stuff",
    "here we have a patent about electrical engineering materials",
]

train_labels = [
    0, # chemistry
    0, # chemistry
    1, # ee
]

# create dummy test pickle
test_texts = [
    "this is a testing patent claim about some chemistry stuff",
    "this is another testing patent claim about some chemistry stuff",
    "here we have a testing patent about electrical engineering materials",
]

test_labels = [
    0, # chemistry
    0, # chemistry
    1, # ee
]

labels_ID = {
    "chemistry": 0,
    "electrical_engineering": 1,
}

save_data("train", train_texts, train_labels, labels_ID)
save_data("test", test_texts, test_labels, labels_ID)
