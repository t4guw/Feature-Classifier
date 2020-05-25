#!/usr/bin/env python

'''
Parses solutions.txt and statements.txt.
Creates problem objects for each LeetCode problem that are serialized as JSON.
Produces JSON file that is intended to be the input of the labeling functions
5/27/2020
'''

import json
from problem import Problem
from itertools import islice

OUTPUT_FILE="problems.json"
PROBLEMS = {}

'''
Example solution:
    ~~~START_SOLUTION~~~
    <problem_number>
    public int{} ... 
    ....
    }
    ~~~END_SOLUTION~~~
'''
def process_solutions(filename):
    with open(filename) as f:
        while line := f.readline():
            if "~~~START_SOLUTION~~~" in line:
                continue
            else:
                number = int(line)
                solution = read_section(f, "~~~END_SOLUTION~~~")
                PROBLEMS[number].solutions.append(solution)


'''
Example statement:
    ~~~STATEMENT_START~~~
    1
    Given an array of integers....
'''
def process_statements(filename):
    with open(filename) as f:
        while line := f.readline():
            number = int(line)
            statement = read_section(f, "~~~STATEMENT_START~~~")
            PROBLEMS[number] = Problem(number, statement)
            

def read_section(file, marker=''):
    content = str()
    while line := file.readline():
        if marker in line:
            return content
        else:
            content += line

def create_output_file():
    with open(OUTPUT_FILE, 'w') as outfile:
        for k, v in PROBLEMS.items():
            print(v.serialize(), file = outfile)
            
def main():
    print("Processing problem statements...")
    process_statements("src/data/statements.txt")
    print("Processing solutions...")
    process_solutions("src/data/solutions.txt")
    create_output_file()

if __name__ == "__main__":
    main()