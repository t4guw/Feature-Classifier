
# Labeling functions go here
import json


    
# Returns a list of map labels
# Labels: 0 = no map in solution, >0 = map exists in solution
# Each index in all_labels labels the corresponding index in all_problems
def maps(all_problems):
    all_labels = []
    for problem in all_problems:
        label = 0
        for solution in problem['solutions']:
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
