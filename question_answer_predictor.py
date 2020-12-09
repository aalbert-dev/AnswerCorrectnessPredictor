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
        data = {}; headers = []; count = 0; line_count_limit = 10000
        for row in f:
            row = row.replace('\n', '')
            split_row = row.split(',')
            if count == 0:
                headers = split_row
            else:
                inner_count = 0
                for header in headers:
                    if header in data:
                        data[header].append(split_row[inner_count])
                    else:
                        data[header] = []
                    inner_count += 1
            count += 1
            if count > line_count_limit: break
        return data


"""
Get the complete dataset as a dictionary.
Return the dictionary with questions, lectures, and an example test.
"""
def get_dataset():
    return {'train'         : get_raw_file('train.csv'),
            'questions'     : get_raw_file('questions.csv'),
            'lectures'      : get_raw_file('lectures.csv'),
            'example_test'  : get_raw_file('example_test.csv')}


# questions = get_raw_file('questions.csv')
# num_questions = len(questions['question_id'])
# for i in range(0, num_questions):
#     question_id = questions['question_id'][i]
#     bundle_id = questions['bundle_id'][i]
#     correct_answer = questions['correct_answer'][i]
#     part = questions['part'][i]
#     tags = questions['tags'][i]
#     #print(question_id, bundle_id, correct_answer, part, tags)
#     print(correct_answer)
#     #input()
print('loading data...')
training_data = get_raw_file('train.csv')
print(training_data.keys())
    