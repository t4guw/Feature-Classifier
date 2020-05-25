from embedder import get_embeddings
from modeldata import ModelData
import labeler as Label
import random
import json
from sklearn import svm


# Creates a list of dictionaries containing problems of the form:
# { 
#   number : <number>,
#   statement : "baojgiodfjiodf",
#   solution  : [ iujgiodjgdiofgj, difjgdfiogjiofgjdiofg, dkfopdfkgopdfkgo, ..., ifjgiofdjgo ]
# }

def read_problems():
    all_problems = []
    with open('problems.json') as f:
        for line in f:
            problem = json.loads(line)
            all_problems.append(problem)
    return all_problems


all_problems = read_problems()
print('\n\n\n', all_problems[0]['solutions'], '\n\n\n')

map_labels = Label.maps(all_problems)
list_labels = []
loop_labels = []
nested_loop_labels = []

print(map_labels)
print(len(map_labels))

all_embeddings = get_embeddings(all_problems)
print('\nEmbeddings Length:', len(all_embeddings), '\n')

data = ModelData(all_embeddings, map_labels)

print(len(data.training_embeddings))
print(len(data.training_labels))
print(len(data.validation_embeddings))
print(len(data.validation_labels))

print('\nFitting SVM...\n')

svm = svm.SVC(kernel='rbf')
svm.fit(data.training_embeddings, data.training_labels)

print('\nValidating SVM...\n')
correct = 0
total = len(data.validation_labels)
for i in range(total):
    if svm.predict([data.validation_embeddings[i]])[0] == data.validation_labels[i]:
        correct += 1

print('\nAccuracy:', correct / total)
