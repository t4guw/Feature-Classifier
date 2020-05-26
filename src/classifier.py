from embedder import get_embeddings
from modeldata import ModelData
import labeler as Label
import random
import json
from sklearn import svm
import numpy as np

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
list_labels = Label.lists(all_problems)
loop_labels = []
nested_loop_labels = []

print(list_labels)
print(len(list_labels))
print('Positive:', list_labels.count(1), '/ 763')

all_embeddings = get_embeddings(all_problems)
print('\nEmbeddings Length:', len(all_embeddings), '\n')


def gridsearch(labels, random_trials, c_vals, g_vals):
    max_correct = 0
    max_c = 0
    max_g = 0
    
    for trial in range(random_trials):
        print('Trial:', trial + 1)
        data = ModelData(all_embeddings, labels)
        total = len(data.validation_labels)
        for c in c_vals:
            for gamma in g_vals:
                correct = 0
                my_svm = svm.SVC(kernel='rbf', C=c, gamma=gamma)
                my_svm.fit(data.training_embeddings, data.training_labels)
    
                for i in range(total):
                    if my_svm.predict([data.validation_embeddings[i]])[0] == data.validation_labels[i]:
                        correct += 1
                if correct > max_correct:
                    max_correct = correct
                    max_c = c
                    max_g = gamma
    
    print('\nAccuracy:', max_correct / total)
    print('C =', max_c, '\ngamma =', max_g)


c_range = np.logspace(-3, 3, 50)
g_range = np.logspace(-3, 3, 50)

print('\n\nOptimizing SVM hyperparameters for lists...')
gridsearch(list_labels, 10, c_range, g_range)

print('\n\nOptimizing SVM hyperparameters for maps...')
gridsearch(map_labels, 10, c_range, g_range)