
# Labeling functions go here
import json

PROBLEMS = [] 


# Creates a list of dictionaries containing problems of the form:
# { 
#   number : <number>,
#   statement : "baojgiodfjiodf",
#   solution  : [ iujgiodjgdiofgj, difjgdfiogjiofgjdiofg, dkfopdfkgopdfkgo, ..., ifjgiofdjgo ]
# }

def read_problems():
    with open('problems.json') as f:
        for line in f:
            problem = json.loads(line)
            print(json.dumps(problem, indent=4, sort_keys=True)) # Printed for demonstrative purposes. Feel free to remove.
            PROBLEMS.append(problem)

    
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
    
def main():
    read_problems()

if __name__ == '__main__':
    main()

