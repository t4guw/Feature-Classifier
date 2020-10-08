
# Labeling functions go here
import json
import re
from itertools import islice
from problem import Problem

 
# Returns a list of map labels
# Labels: 0 = no map in solution, >0 = map exists in solution
# Each index in all_labels labels the corresponding index in all_problems
def maps(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem.solutions:
            for line in solution.split('\n'):
                if 'map<' in line.lower():
                    label = 1
        all_labels.append(label)
    return all_labels


def lists(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem['solutions']:
            for line in solution.split('\n'):
                if 'list<' in line.lower() or 'vector<' in line.lower():
                    label = 1
        all_labels.append(label)
    return all_labels


def sets(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem['solutions']:
            for line in solution.split('\n'):
                if 'set<' in line.lower():
                    label = 1
        all_labels.append(label)
    return all_labels


def stacks(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem['solutions']:
            for line in solution.split('\n'):
                if 'stack<' in line.lower():
                    label = 1
        all_labels.append(label)
    return all_labels


def queues(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem['solutions']:
            for line in solution.split('\n'):
                if 'queue<' in line.lower():
                    label = 1
        all_labels.append(label)
    return all_labels


def priority_queues(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem['solutions']:
            for line in solution.split('\n'):
                if 'priority_queue<' in line.lower() or 'priorityqueue<' in line.lower():
                    label = 1
        all_labels.append(label)
    return all_labels


def loops(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem['solutions']:
            for line in solution.split('\n'):
                for keywords in ['for', 'do', 'while']:
                    if re.search(r'\b' + keywords + r'\b', line.lower()):
                        label = 1
        all_labels.append(label)
    return all_labels


def switches(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem['solutions']:
            for line in solution.split('\n'):
                if re.search(r'\bswitch\b', line.lower()):
                    label = 1
        all_labels.append(label)
    return all_labels


def if_statements(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem['solutions']:
            for line in solution.split('\n'):
                if re.search(r'\bif\b', line.lower()):
                    label = 1
        all_labels.append(label)
    return all_labels



# problem_list = process_solutions('/Users/victor/College/Fall-Quarter/Research/Feature-Classifier/src/clean/solutions.txt')
# for s in problem_list:
#     print(s)


