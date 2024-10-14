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
import time
from AutoEncoder import autoEncoder
from loglizer import dataloader, preprocessing

struct_log = 'data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = 'data/HDFS/anomaly_label.csv' # The anomaly label file
max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.3 # the threshold for anomaly detection
def menu():
    print("\nChose an Option:")
    print("1. Event Mappings")
    print("2. Anomaly Description")
    print("3. Exit")

    choice = input("Enter the number of your choice: ").strip()

    if choice == '1':
        return "1"
    elif choice == '2':
        return "2"
    elif choice == "3":
        return "exit"


if __name__ == '__main__':
    print("------------LOG ADVISOR------------\n\n")
    print("Injecting data from HDFS....\n")
    (x_train, y_train), (x_test, y_test), (event_dict) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test2 = feature_extractor.transform(x_test)

    print("Observing log data for anomalies.....\n")

    #sleep
    print("\tIsolation Forest Model executing...")
    y_out1 = isolationForest(x_train=x_train, x_test=x_test2)

    print("\tComplete.\n")
    print("\tLog Clustering Model executing...")
    y_out2 = logClustrering(x_train=x_train,y_train=y_train, x_test=x_test2)
    print("\tComplete.\n")
    print("\tAuto Encoder executing...")
    y_out3 = autoEncoder(x_test=x_test2,x_train=x_train,y_train=y_train)
    print("\tComplete.\n")

    res = []

    for i in range(len(x_test)):
        if y_out2[i]+y_out1[i]+y_out3[i]>=2:
            res.append(x_test[i])

    # print(y_out1)
    # print(y_out2)
    # print(y_out3)
    # print(res, len(res))
    total_blocks=len(x_test)
    anomalous_blocks =len(res)

    print("Summary:")
    print(f"\tTotal blocks: {total_blocks}\n\tAnomalous blocks: {anomalous_blocks}")

    opt = input("\nShow events related to anomalous blocks? (Y/N) ")

    if opt == "Y" or opt == "y":
        count =1
        for i in res:
            print(count, end=". ")
            for j in i:
                print(j, end=" ")
            print()
            count+=1


    while True:
        flag = menu()
        if flag == "exit":
            break
        elif flag == "1":
            for i, j in event_dict.items():
                print(f"{i}:\t{j}")
        elif flag =="2":
            inp = int(input("Enter sequence number: "))

            response = requests.post('http://192.168.110.52:5000/gemini', json={'sequence': res[inp-1]})
            print("Anomaly Description:\n")
            # Print the server's response
            if response.status_code == 200:
                print(response.json()['response'])
            else:
                print('Error:', response.status_code)

    
    # while True:
    #     flag = menu()
    #     if flag == "exit":
    #         break
    #     elif flag == "gemini":
    #         url = 'http://192.168.110.52:5000/gemini'  # Flask endpoint for Gemini
    #     else:
    #         url = 'http://192.168.110.52:5000/llama'
    #     # Send POST request to Flask server
    #     response = requests.post(url, json={'sequence': sequence})

    #     # Print the server's response
    #     if response.status_code == 200:
    #         print('Response from server:', response.json()['response'])
    #     else:
    #         print('Error:', response.status_code)


   
