
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
    