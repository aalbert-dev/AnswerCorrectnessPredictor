import csv, sys, os, numpy, math, decimal, copy, random
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
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
def get_raw_file(fname, line_count_limit):
    with open(os.path.join(sys.path[0], fname),'r') as f:
        data = {}; headers = []; count = 0
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
Extract features from a given row in the training data.
Return the row as a list of integer features.
"""
def get_row_features(row, train_data, feature_names):
    return [int(train_data[feature_name][row]) for feature_name in feature_names]


"""
Convert the raw training data into lists of features.
Return the training data as a list of lists containing integer features.
"""
def extract_features(train_data):
    features = []; data_len = len(train_data['row_id'])
    for i in range(0, data_len):
        feature_names = ['user_id', 'content_id']
        row_feature_values = get_row_features(i, train_data, feature_names)
        features.append(row_feature_values)
    return features


"""
Get the labels from the training data as integer labels.
Return the list of integer labels for the training data.
"""
def extract_labels(train_data):
    data_len = len(train_data['row_id'])
    return [int(train_data['answered_correctly'][i]) for i in range(0, data_len)]


"""
Attempt at using the LGBM model for knowledge tracing task.
Checks if model has already been trained to save time and then
either trains and returns a new model or returns the already 
trained model.
"""
def get_trained_lgb_model(model_name):
    model_path = os.path.join(sys.path[0], model_name)
    if os.path.exists(model_path):
        return lgb.Booster(model_file=model_path)
    else:
        training_data = get_raw_file('train.csv')
        features = numpy.asarray(extract_features(training_data))
        labels = numpy.asarray(extract_labels(training_data))
        train_data = lgb.Dataset(features, label=labels, feature_name=['user_id', 'content_id'], categorical_feature=['user_id', 'content_id'])
        bst = lgb.train({}, train_data)
        bst.save_model(model_path)
        return bst


"""

"""
def split_data(all_data):
    data_by_user = {}; data_len = len(all_data['row_id'])
    for i in range(0, data_len):
        this_user_id = all_data['user_id'][i]
        row_features = get_row_features(i, all_data, ['user_id', 'content_id'])
        row_label = all_data['answered_correctly'][i]
        if this_user_id in data_by_user:
            data_by_user[this_user_id].append((row_features, row_label))
        else:
            data_by_user[this_user_id] = []
            data_by_user[this_user_id].append((row_features, row_label))

    training_data = []; training_labels = []; testing_data = []; testing_labels = []
    for user_id in data_by_user.keys():
        user_data_Len = len(data_by_user[user_id])
        split_index = int((user_data_Len - 1) / 4 * 3); count = 0
        for features, label in data_by_user[user_id]:
            if count <= split_index:
                training_data.append(features)
                training_labels.append(label)
            else:
                testing_data.append(features)
                testing_labels.append(label)
            count += 1
    return training_data, training_labels, testing_data, testing_labels


def calculate_score(results, labels):
    num_correct = 0; num_incorrect = 0
    combined_results = zip(results, labels)
    for result, label in combined_results:
        if result != label:
            num_incorrect += 1
        else:
            num_correct += 1
    score = num_correct / (num_correct + num_incorrect)
    return score 

training_data = get_raw_file('train.csv', 1000000)
training_data, training_labels, testing_data, testing_labels, = split_data(training_data)

classifier = RandomForestClassifier(n_estimators=5000, random_state=0)
classifier.fit(training_data, training_labels)


results = classifier.predict(testing_data)

score = calculate_score(results, testing_labels)

print(score)