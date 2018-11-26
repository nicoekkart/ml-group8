# standard packages used to handle files
import sys
import os
import glob
import time

# commonly used library for data manipilation
import pandas as pd

# numerical
import numpy as np

# handle images - opencv
import cv2

# machine learning library
import sklearn
import sklearn.preprocessing

#used to serialize python objects to disk and load them back to memory
import pickle

#plotting
import matplotlib.pyplot as plt

# helper functions kindly provided for you by Matthias 
import helpers
# specific helper functions for feature extraction
import features


# filepath constants
DATA_BASE_PATH = './'
OUTPUT_PATH='./'


DATA_TRAIN_PATH = os.path.join(DATA_BASE_PATH,'train')
DATA_TEST_PATH = os.path.join(DATA_BASE_PATH,'test')

FEATURE_BASE_PATH = os.path.join(OUTPUT_PATH,'features')
FEATURE_TRAIN_PATH = os.path.join(FEATURE_BASE_PATH,'train')
FEATURE_TEST_PATH = os.path.join(FEATURE_BASE_PATH,'test')

PREDICTION_PATH = os.path.join(OUTPUT_PATH,'predictions')

# filepatterns to write out features
FILEPATTERN_DESCRIPTOR_TRAIN = os.path.join(FEATURE_TRAIN_PATH,'train_features_{}.pkl')
FILEPATTERN_DESCRIPTOR_TEST = os.path.join(FEATURE_TEST_PATH,'test_features_{}.pkl')

# create paths in case they don't exist:
helpers.createPath(FEATURE_BASE_PATH)
helpers.createPath(FEATURE_TRAIN_PATH)
helpers.createPath(FEATURE_TEST_PATH)
helpers.createPath(PREDICTION_PATH)



import yaml
with open(sys.argv[1]) as config_file:
    config = yaml.load(config_file)
print(config)


# TODO: Maybe try normalization
folder_paths = glob.glob(os.path.join(DATA_TRAIN_PATH,'*'))
label_strings = np.sort(np.array([os.path.basename(path) for path in folder_paths]))
num_classes = label_strings.shape[0]
print(label_strings)
print('Number of classes:', num_classes)


train_paths = dict((label_string, helpers.getImgPaths(os.path.join(DATA_TRAIN_PATH,label_string))) for label_string in label_strings)



descriptor_desired='sift_' + str(config['feature_count'])
with open(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired),'rb') as pkl_file_train:
    ## read
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(FILEPATTERN_DESCRIPTOR_TRAIN.format(descriptor_desired))
    
    for _ in range(0, input_size, max_bytes):
        bytes_in += pkl_file_train.read(max_bytes)
    train_features_from_pkl = pickle.loads(bytes_in)


    
print('Number of encoded train images: {}'.format(len(train_features_from_pkl)))


# CONSTRUCT CODEBOOK

# learn the codebook for the 'freak' features from the training data

clustered_codebook = helpers.createCodebook(train_features_from_pkl, codebook_size = int(config['codebook_size'] * config['feature_count']))

# encode all train images 
train_data=[]
train_labels_old=[]

print('Completed clustering codebook')
for i, image_features in enumerate(train_features_from_pkl):
    bow_feature_vector = helpers.encodeImage(image_features.data,clustered_codebook)
    train_data.append(bow_feature_vector)
    train_labels_old.append(image_features.label)


# use a labelencoder to obtain numerical labels
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(label_strings)
print(train_labels_old[:10])
train_labels = label_encoder.transform(train_labels_old)
print(train_labels)




from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import cross_val_score

simple_pipe = make_pipeline(SVC(random_state=0, probability=True, kernel='linear',C=config['regularization']))
   # LogisticRegression(class_weight='balanced', random_state=0, tol=1e-5, C=2))

scores1 = cross_val_score(simple_pipe, train_data, train_labels, cv=5, scoring='neg_log_loss')
print(scores1)
print("Average validation accuracy: ",scores1.mean(),", stdev: ",scores1.std())
