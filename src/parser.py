#!/usr/bin/env python

'''
Parses solutions.txt and statements.txt.
Creates problem objects for each LeetCode problem that are serialized as JSON.
Produces JSON file that is intended to be the input of the labeling functions
5/27/2020
'''

import json

OUTPUT_NAME="problems.json"
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
def read_statements(filename):
    
    with open(filename) as f:
        read_data = f.read()
    pass


'''
Example statement:
    ~~~STATEMENT_START~~~
    1
    Given an array of integers....
'''

def read_solutions(filename):
    pass



def main():
    pass

if __name__ == "__main__":
    pass