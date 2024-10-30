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
from dynamo_preprocess2 import generate_anomalous_sequences, generate_non_anomalous_sequences
import time
from auto_encoder import predict
from random_forest import predict2
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import numpy as np
from loglizer import dataloader, preprocessing
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
    print("Fetching logs....\n")
    non_an = generate_non_anomalous_sequences(500)
    an = generate_anomalous_sequences(30)
    x_test = np.array(non_an+an)

    event_dict = {
        "E1": "CreateTable",
        "E2": "PutItem",
        "E3": "UpdateItem",
        "E4": "DeleteTable"
    }

    with open("nonan.txt", 'w') as file:
        for data in non_an:
            file.write(str(data) + '\n') 
    with open("an.txt", 'w') as file:
        for data in an:
            file.write(str(data) + '\n') 

    print("Observing log data for anomalies.....\n")

    #sleep
    # print("\tIsolation Forest Model executing...")
    # y_out1 = isolationForest(x_train=x_train, x_test=x_test2)

    # print("\tComplete.\n")
    # print("\tLog Clustering Model executing...")
    # y_out2 = logClustrering(x_train=x_train, x_test=x_test2)
    # print("\tComplete.\n")
    print("Analyzing event sequences using Random Forest...")
    y_out2, non_an = predict2(x_test)
    print("\tComplete")
    print(f"\t{len(y_out2)} anomalous sequences detected.\n")



    print("Analyzing remaining event sequences using autoencoder...")
    y_out3 = predict(non_an)
    print("\tComplete")
    print(f"\t{len(y_out3)} anomalous sequences detected.\n")

    res = y_out2+y_out3

    # print(y_out1,y_out2,y_out3)

    # for i in range(len(x_test)):
    #     # if y_out2[i]+y_out1[i]+y_out3[i]>=1:
    #     if y_out1[i]==1:
    #         res.append(x_test[i])

    # print(y_out1)
    # print(y_out2)
    # print(y_out3)
    # print(res, len(res))
    total_blocks=len(x_test)
    anomalous_blocks =len(res)
    
    print("Summary:")
    print(f"\tTotal Event Sequences: {total_blocks}\n\tAnomalous Sequences: {anomalous_blocks}")

    opt = input("\nShow anomalous sequences? (Y/N) ")

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
            print("\nAnomaly Description:\n\n", res[inp-1], "\n")
            # Print the server's response
            if response.status_code == 200:
                print(response.json()['response'])
            else:
                print('Error:', response.status_code)