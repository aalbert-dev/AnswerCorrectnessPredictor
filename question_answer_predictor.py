import riiideducation # enviorment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb # light gradient boosting model
import dask.dataframe as dd # dask fast dataframe importer
from sklearn.metrics import roc_auc_score # area under the curve calculator
from sklearn.preprocessing import LabelEncoder # label encoder
import matplotlib.pyplot as plt # plotting

import gc 
import os 
import sys


training_path           = '/kaggle/input/riiid-test-answer-prediction/train.csv'
questions_path          = '/kaggle/input/riiid-test-answer-prediction/questions.csv'
lectures_path           = '/kaggle/input/riiid-test-answer-prediction/lectures.csv'
example_submission_path = '/kaggle/input/riiid-test-answer-prediction/example_sample_submission.csv'
example_test_path       = '/kaggle/input/riiid-test-answer-prediction/example_test.csv'

"""
Print the paths for local enviorment input files.
"""
def print_paths():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            

"""
Print the input string to the notebook console. 
Also prints the input string to Kaggle kenrel.
"""
def print_log(s):
    print(s)
    sys.__stdout__.write(s)
    

"""
Convert the existing dataframe values to data types.
Also replace NaN values with -1. 
Returns the dataframes for training as well as labels.
"""    
def getDataFramesForTraining(dataframe):
    data = dataframe[features]
    data['prior_question_elapsed_time'].fillna(-1, inplace=True)
    return data, dataframe['answered_correctly']


"""
Convert the existing dataframe values to data types.
Also replace NaN values with -1. 
Returns the dataframes for testing.
"""
def getDataFramesForTesting(dataframe):
    data = dataframe[features]
    data['prior_question_elapsed_time'].fillna(-1, inplace=True)
    return data


"""
Convert the existing dataframe values to correct data types.
Splice tag column into multiple columns and append them to questions dataframe.
Returns the dataframes for questions.
"""
def getDataFramesForQuestions(dataframe):
    tag = dataframe["tags"].str.split(" ", n = 10, expand = True) 
    tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']
    dataframe =  pd.concat([dataframe,tag], axis=1)
    for tag in tag.columns:
        dataframe[tag] = pd.to_numeric(dataframe[tag], errors='coerce')
    return dataframe

"""
Merge in the questions tag columns joining
on question_id and content_id.
Return the merged dataframe.
"""
def mergeQuestions(train, questions):
    return pd.merge(train, questions, left_on = 'content_id', right_on = 'question_id', how = 'left') 


"""

"""
def encodeColumns(dataframe):
    lb_make = LabelEncoder()
    dataframe['prior_question_had_explanation'].fillna(True, inplace=True)
    dataframe["prior_question_had_explanation_enc"] = lb_make.fit_transform(dataframe["prior_question_had_explanation"])
    return dataframe


"""

"""
def expandTags(questions):
    tag = questions["tags"].str.split(" ", n = 10, expand = True) 
    tag.columns = ['tags1','tags2','tags3','tags4','tags5','tags6']
    questions =  pd.concat([questions,tag],axis=1)
    for tag in tag.columns:
        questions[tag] = pd.to_numeric(questions[tag], errors='coerce')
    return questions


"""

"""
def plot_metrics(model, eval_results):
    lgb.plot_importance(model)
    plt.show()
    metrics = ['auc', 'l1', 'l2']
    for metric in metrics: 
        lgb.plot_metric(eval_results, metric=metric)
        plt.show()
        

"""

"""
def createSubmission(model):
    env = riiideducation.make_env()
    for test_df, sample_prediction_df in env.iter_test():
        test_df_enc = encodeColumns(test_df)
        test_df_enc = mergeQuestions(test_df_enc, questions)
        testdata = getDataFramesForTesting(test_df_enc)
        test_df['answered_correctly'] =  model.predict(testdata[features])
        env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
        

"""

"""
def splitTrainValid(train, labels):
    trainingCount = int(len(labels.index) * training_valid_ratio)
    train_dataset = lgb.Dataset(train[:trainingCount], labels[:trainingCount], categorical_feature = categorical_features)
    valid_dataset = lgb.Dataset(train[trainingCount:], labels[trainingCount:], categorical_feature = categorical_features)
    return train_dataset, valid_dataset

    
features = ['content_id', 'prior_question_elapsed_time', 'prior_question_had_explanation_enc', 'tags1','tags2','tags3','tags4']
categorical_features = ['tags1','tags2','tags3','tags4']

training_cols = ['content_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'answered_correctly']
question_cols = [0, 1, 3, 4]

training_dtype = {'content_id': 'int16','content_type_id': 'int8','answered_correctly':'int8','prior_question_elapsed_time': 'float32','prior_question_had_explanation': 'boolean'}
question_dtype = {'question_id': 'int16', 'part': 'int8','bundle_id': 'int8','tags': 'str'}

million = 1000000
training_valid_ratio = 0.9
max_iterations = 200
eval_round = 10
eval_results = {}

training_params = {'objective'            : 'binary',
                   'metric'               :('auc', 'l1', 'l2'),
                   'boosting'             : 'gbdt',
                   'tree_learner'         : 'voting',
                   'learning_rate'        :  0.11,
                   'num_leaves'           :  80,
                   'min_data_in_leaf'     :  20,
                   'early_stopping_rounds':  10}


train = pd.read_csv(training_path, nrows=50*million, engine='c', usecols=training_cols, dtype=training_dtype)

gc.collect()
print_log('Done reading training data.')

questions = pd.read_csv(questions_path, usecols=question_cols, dtype=question_dtype)

gc.collect()
print_log('Done reading questions data.')

questions = expandTags(questions)

gc.collect()
print_log('Done grabbing tags.')

train = mergeQuestions(train, questions)

gc.collect()
print_log('Done merging questions.')

train = encodeColumns(train)

gc.collect()
print_log('Done encoding columns.')

train, labels = getDataFramesForTraining(train)

gc.collect()
print_log('Done labelling data.')

train_dataset, valid_dataset = splitTrainValid(train, labels)

gc.collect()
print_log('Done splitting training and valid.')

model = lgb.train(training_params, train_dataset, valid_sets=[train_dataset, valid_dataset], num_boost_round=max_iterations, verbose_eval=eval_round, evals_result=eval_results)

gc.collect()
print_log('Done training.')

createSubmission(model)
    
gc.collect()    
print_log('Done with inference.')

plot_metrics(model, eval_results)

gc.collect()
print_log('Done all.')