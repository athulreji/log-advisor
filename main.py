#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import os
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import requests
from IsolationForest import isolationForest
from LogClustering import logClustrering
from AutoEncoder import autoEncoder
from loglizer import dataloader, preprocessing

struct_log = 'data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = 'data/HDFS/anomaly_label.csv' # The anomaly label file
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.3 # the threshold for anomaly detection

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test2 = feature_extractor.transform(x_test)

    y_out1 = isolationForest(x_train=x_train, x_test=x_test2)
    y_out2 = logClustrering(x_train=x_train,y_train=y_train, x_test=x_test2)
    y_out3 = autoEncoder(x_test=x_test2,x_train=x_train,y_train=y_train)

    res = []

    for i in range(len(x_test)):
        if y_out2[i]+y_out1[i]+y_out3[i]>=2:
            res.append(x_test[i])

    # print(y_out1)
    # print(y_out2)
    # print(y_out3)
    # print(res, len(res))

    sequence = res[0]

# Send POST request to Flask server
    response = requests.post('http://192.168.110.52:5000/gemini', json={'sequence': sequence})

    # Print the server's response
    if response.status_code == 200:
        print('Response from server:', response.json()['response'])
    else:
        print('Error:', response.status_code)


