from embedder import get_embeddings
from modeldata import ModelData
import labeler as Label
import random
from sklearn import svm

# This will be the list of Problem objects:
# =========================================
all_problems = []
# =========================================

# These will be the list of labels called like so:
#
# map_labels = Label.maps(all_problems)
# list_labels = Label.loops(all_problems)
# ...
# =================================
map_labels = []
list_labels = []
loop_labels = []
nested_loop_labes = []
# ... and so on
# =================================


all_embeddings = get_embeddings(all_problems)
data = ModelData(all_embeddings, map_labels)

svm = svm.SVC(kernel='rbf')
svm.fit(data.training_embeddings, data.training_labels)

correct = 0
total = len[data.validation_labels]
for i in range(total):
    if svm.predict(data.validation_embeddings[i])[0] == data.validation_labels[i]:
        correct += 1

print('Accuracy:', correct / total)
