
# Returns a list of map labels
# Labels: 0 = no map in solution, >0 = map exists in solution
# Each index in all_labels labels the corresponding index in all_problems
def maps(all_problems):
    all_labels = []
    label = 0
    for problem in all_problems:
        for solution in problem.solutions:
            for line in solution:
                if 'map<' in line:
                    label += 1
        all_labels.append(label)
    return all_labels

# Returns a list of vector labels
# Labels: 0 = no vector in solution, >0 = vector exists in solution
# Each index in all_labels labels the corresponding index in all_problems
def vectors(all_problems):
    all_labels = []
    label = 0
    for problem in all_problems:
        for solution in problem.solutions:
            for line in solution:
                if 'vector<' in line:
                    label += 1
        all_labels.append(label)
    return all_labels

# Returns a list of set labels
# Labels: 0 = no set in solution, >0 = set exists in solution
# Each index in all_labels labels the corresponding index in all_problems
def sets(all_problems):
    all_labels = []
    label = 0
    for problem in all_problems:
        for solution in problem.solutions:
            for line in solution:
                if 'set<' in line:
                    label += 1
        all_labels.append(label)
    return all_labels

# Returns a list of stack labels
# Labels: 0 = no stack in solution, >0 = stack exists in solution
# Each index in all_labels labels the corresponding index in all_problems
def stacks(all_problems):
    all_labels = []
    label = 0
    for problem in all_problems:
        for solution in problem.solutions:
            for line in solution:
                if 'stack<' in line:
                    label += 1
        all_labels.append(label)
    return all_labels

# Returns a list of queue labels
# Labels: 0 = no queue in solution, >0 = queue exists in solution
# Each index in all_labels labels the corresponding index in all_problems
def queues(all_problems):
    all_labels = []
    label = 0
    for problem in all_problems:
        for solution in problem.solutions:
            for line in solution:
                if 'queue<' in line:
                    label += 1
        all_labels.append(label)
    return all_labels
