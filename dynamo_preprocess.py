import json

import numpy as np
# Open and read the JSON file

def preprocess():
    with open("non_anomalous_dynamodb_logs.json", "r") as file1:
        data1 = json.load(file1)  # Load the JSON data


    tables = {}

    sources = {}

    requests = {}


    eventMap = {}
    count = 1
    for record in data1["Records"]:
        if record['eventName']  not in eventMap:
            eventMap[record['eventName']] = "E"+ str(count)
            count+=1
        eventID = eventMap[record['eventName']]

        if record["requestParameters"]["tableName"] not in tables:
            tables[record["requestParameters"]["tableName"]] = []
        tables[record["requestParameters"]["tableName"]].append(eventID)

        if record["requestParameters"]["tableName"] + record["sourceIPAddress"] not in requests:
            requests[record["requestParameters"]["tableName"] + record["sourceIPAddress"]] = []
        requests[record["requestParameters"]["tableName"] + record["sourceIPAddress"]].append(eventID)

        if record["sourceIPAddress"] not in sources:
            sources[record["sourceIPAddress"]] = []
        sources[record["sourceIPAddress"]].append(eventID)

    # 
    
    x_train = np.array([list(j) for i,j in tables.items()])

    with open("anomalous_dynamodb_logs.json", "r") as file2:
        data2 = json.load(file2)  # Load the JSON data


    tables = {}

    sources = {}

    requests = {}
    for record in data2["Records"]:
        eventID = eventMap[record['eventName']]

        if record["requestParameters"]["tableName"] not in tables:
            tables[record["requestParameters"]["tableName"]] = []
        tables[record["requestParameters"]["tableName"]].append(eventID)

        if record["requestParameters"]["tableName"] + record["sourceIPAddress"] not in requests:
            requests[record["requestParameters"]["tableName"] + record["sourceIPAddress"]] = []
        requests[record["requestParameters"]["tableName"] + record["sourceIPAddress"]].append(eventID)

        if record["sourceIPAddress"] not in sources:
            sources[record["sourceIPAddress"]] = []
        sources[record["sourceIPAddress"]].append(eventID)

    # 
    
    x_test = np.array([list(j) for i,j in tables.items()])
    for i, j in eventMap.items():
        print(j,": ", i)
    return x_train, x_test