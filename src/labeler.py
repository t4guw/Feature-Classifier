
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

def main():
    read_problems()

if __name__ == '__main__':
    main()