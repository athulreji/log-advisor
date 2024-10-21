import json

# Open and read the JSON file

def preprocess():
    with open("non_anomalous_dynamodb_logs.json", "r") as file:
        data = json.load(file)  # Load the JSON data


    tables = {}

    sources = {}

    requests = {}


    eventMap = {}
    count = 1
    for record in data["Records"]:
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

    for i,j in eventMap.items():
        print(i,":\t", j)


    print("Tables",tables, end="\n\n\n\n")
    print("Sources", sources, end="\n\n\n\n")
    print("Requests", requests)
