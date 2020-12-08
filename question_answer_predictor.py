import csv, sys, os, numpy, math, decimal, copy, random
"""
@author     Arjun Albert, Henry Arvans 
@email      arjunalbert@brandeis.edu, harvans5@brandeis.edu
@modified   12/8/2020
@notes      Kaggle answer correctness predictor for COSI123A final project.
"""

"""
Retrive a csv file relative to where the program was run.
Returns the csv file as a dictionary.
"""
def get_raw_file(fname):
    with open(os.path.join(sys.path[0], fname),'r') as f:
        data = {}; headers = []; count = 0
        for row in f:
            row = row.replace('\n', '')
            split_row = row.split(',')
            if count == 0:
                headers = split_row
                count += 1
            else:
                inner_count = 0
                for header in headers:
                    if header in data:
                        data[header].append(split_row[inner_count])
                    else:
                        data[header] = []
                    inner_count += 1
        return data


questions = get_raw_file('questions.csv')

lectures = get_raw_file('lectures.csv')

example_test = get_raw_file('example_test.csv')