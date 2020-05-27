from embedder import get_embeddings, get_embeddings_from_file
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
# print('\n\n\n', all_problems[0]['solutions'], '\n\n\n')

map_labels = Label.maps(all_problems)
list_labels = Label.lists(all_problems)
set_labels = Label.sets(all_problems)
stack_labels = Label.stacks(all_problems)
queue_labels = Label.queues(all_problems)
priority_queue_labels = Label.priority_queues(all_problems)

loop_labels = []
nested_loop_labels = []

# print(list_labels)


# exit()

all_embeddings = get_embeddings_from_file()
print('Single Embedding:', len(all_embeddings[0]))
print('\nEmbeddings Length:', len(all_embeddings), '\n')

data = ModelData(all_embeddings, list_labels)
data.set_data_kfold(10, 0)
print(data.training_labels)

def kfold_cross_validation(labels, k, c, gamma, weight): 
    correct = np.zeros(k)
    weights = {0:weight, 1:1.0}

    for index in range(k):
        data = ModelData(all_embeddings, labels)
        data.set_data_kfold(k, index)
        
        my_svm = svm.SVC(kernel='rbf', C=c, gamma=gamma, class_weight=weights)
        my_svm.fit(data.training_embeddings, data.training_labels)
        # print(my_svm.support_vectors_)
        
        for i in range(len(data.validation_labels)):
            # print(my_svm.predict([data.validation_embeddings[i]])[0], end='', flush=True)
            
            if my_svm.predict([data.validation_embeddings[i]])[0] == data.validation_labels[i]:
                correct[index] += 1
        correct[index] /= len(data.validation_labels)

    # print(correct)
    return np.mean(correct)


def gridsearch(labels, random_trials, c_vals, g_vals, weights):
    positive = labels.count(1)
    print(int((positive / 763) * 100) / 100, '% positive, or', positive, '/ 763')
    max_accuracy = 0
    max_c = 0
    max_g = 0
    max_w = 0
    k = 10
    
    count = 1
    for c in c_vals:
        for gamma in g_vals:
            for weight in weights:
                if count % len(weights) == 0:
                    print(count, end=' ', flush=True)

                accuracy = kfold_cross_validation(labels, k, c, gamma, weight)

                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    max_c = c
                    max_g = gamma
                    max_w = weight

                count += 1
    
    print('\nAccuracy:', max_accuracy)
    print('C =', max_c, '\ngamma =', max_g, 'weight =', max_w)

    


c_range = np.logspace(-1, 2, 10)
g_range = np.logspace(-2, 0, 10)
w_range = np.linspace(0, 0.6, 60)
print(w_range)
print('\n\nOptimizing SVM hyperparameters for lists...')
gridsearch(list_labels, 10, c_range, g_range, w_range)

print('\n\nOptimizing SVM hyperparameters for sets...\n')
gridsearch(set_labels, 10, c_range, g_range, w_range)

print('\n\nOptimizing SVM hyperparameters for stacks...')
gridsearch(stack_labels, 10, c_range, g_range, w_range)

print('\n\nOptimizing SVM hyperparameters for queues...')
gridsearch(queue_labels, 10, c_range, g_range, w_range)

print('\n\nOptimizing SVM hyperparameters for priority queues...')
gridsearch(priority_queue_labels, 10, c_range, g_range, w_range)

print('\n\nOptimizing SVM hyperparameters for maps...')
gridsearch(map_labels, 10, c_range, g_range, w_range)
