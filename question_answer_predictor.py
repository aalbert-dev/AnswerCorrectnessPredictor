import csv, sys, os, numpy, math, decimal, copy, random
import lightgbm as lgb
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
        data = {}; headers = []; count = 0; line_count_limit = 100
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


"""

"""
def get_row_features(row, train_data, feature_names):
    return [train_data[feature_name][row] for feature_name in feature_names]


"""

"""
def extract_features(train_data):
    features = []; data_len = len(train_data['row_id'])
    for i in range(0, data_len):
        feature_names = ['user_id', 'content_id']
        row_feature_values = get_row_features(i, train_data, feature_names)
        features.append(row_feature_values)
    return features


"""

"""
def extract_labels(train_data):
    data_len = len(train_data['row_id'])
    return [train_data['answered_correctly'][i] for i in range(0, data_len)]


training_data = get_raw_file('train.csv')
features = extract_features(training_data)
labels = extract_labels(training_data)
print(labels)