import matplotlib.pyplot as plt
from preprocess_data import load_labels
from preprocess_csv_data import categories
from collections import Counter
import numpy as np

labels = load_labels("test_labels.pkl")
labels.extend(load_labels("train_labels.pkl"))
counter = Counter(labels)
counts = [0] * len(categories)
for key, value in counter.items():
    counts[key] = value
index = np.arange(len(categories)) + 0.3
bar_width = 0.4
total = len(labels)

plt.figure()
pps = plt.bar(index, counts, bar_width)
for p in pps:
   height = p.get_height()
   plt.annotate('{}/{}%'.format(height, round(height / total * 100, 1)),
      xy=(p.get_x() + p.get_width() / 2, height),
      xytext=(0, 2), # 3 points vertical offset
      textcoords="offset points",
      ha='center', va='bottom',
      fontsize=8)

plt.xlabel("Patent categories")
plt.ylabel("# samples")
plt.xticks(index, labels=categories)
plt.title('Patent Category Distribution')
plt.show()