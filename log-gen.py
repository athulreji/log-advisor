import json
from datetime import datetime, timedelta
import random

# Function to create non-anomalous DynamoDB logs


def generate_non_anomalous_logs():
    logs = {
        "Records": []
    }

    tables  = [
        "apple", "banana", "orange", "grape", "strawberry", 
        "watermelon", "kiwi", "blueberry", "peach", "mango",
        "pineapple", "apricot", "pear", "cantaloupe", "papaya",
        "cherry", "blackberry", "fig", "pomegranate", "tangerine",
        "lemon", "lime", "coconut", "date", "guava",
        "plum", "nectarine", "raspberry", "dragonfruit", "jackfruit"
    ]

    sources = ['192.168.1.10', '10.0.0.5', '172.16.254.1', '203.0.113.45', '198.51.100.75', '127.0.0.1', '192.0.2.88', '203.0.113.25', '10.1.1.1', '192.168.0.255']

    for i in range(1000):
        user_arn = "arn:aws:iam::123456789012:user/JohnDoe"
        table_name = "table_"+ str(i+1)
        source = random.choice(sources)
        timestamps = [datetime(2024, 10, 11, 12, 45, 20) + timedelta(minutes=i * 10) for i in range(4)]

        logs["Records"].append({
            "eventVersion": "1.0",
            "eventTime": timestamps[0].isoformat() + "Z",
            "eventSource": "dynamodb.amazonaws.com",
            "eventName": "CreateTable",
            "userIdentity": {
                "type": "IAMUser",
                "arn": user_arn
            },
            "requestParameters": {
                "tableName": table_name
            },
            "sourceIPAddress": source,
            "userAgent": "aws-cli/2.0.30"
        })

        for _ in range(random.randint(1,4)):
            for _ in range(random.randint(0,5)):
                logs["Records"].append({
                    "eventVersion": "1.0",
                    "eventTime": timestamps[1].isoformat() + "Z",
                    "eventSource": "dynamodb.amazonaws.com",
                    "eventName": "PutItem",
                    "userIdentity": {
                        "type": "IAMUser",
                        "arn": user_arn
                    },
                    "requestParameters": {
                        "tableName": table_name,
                        "item": {
                            "OrderID": {"S": "12345"},
                            "CustomerID": {"S": "98765"}
                        }
                    },
                    "sourceIPAddress": source,
                    "userAgent": "aws-cli/2.0.30"
                })

            for _ in range(random.randint(0,5)):
                logs["Records"].append({
                    "eventVersion": "1.0",
                    "eventTime": timestamps[2].isoformat() + "Z",
                    "eventSource": "dynamodb.amazonaws.com",
                    "eventName": "UpdateItem",
                    "userIdentity": {
                        "type": "IAMUser",
                        "arn": user_arn
                    },
                    "requestParameters": {
                        "tableName": table_name,
                        "key": {
                            "OrderID": {"S": "12345"}
                        },
                        "updateExpression": "SET OrderStatus = :status",
                        "expressionAttributeValues": {
                            ":status": {"S": "SHIPPED"}
                        }
                    },
                    "sourceIPAddress": source,
                    "userAgent": "aws-cli/2.0.30"
                })

        logs["Records"].append({
            "eventVersion": "1.0",
            "eventTime": timestamps[3].isoformat() + "Z",
            "eventSource": "dynamodb.amazonaws.com",
            "eventName": "DeleteTable",
            "userIdentity": {
                "type": "IAMUser",
                "arn": user_arn
            },
            "requestParameters": {
                "tableName": table_name
            },
            "sourceIPAddress": source,
            "userAgent": "aws-cli/2.0.30"
        })

    return logs

# Function to create anomalous DynamoDB logs
def generate_anomalous_logs():
    logs = {
        "Records": []
    }

    tables  = [
        "apple", "banana", "orange", "grape", "strawberry", 
        "watermelon", "kiwi", "blueberry", "peach", "mango",
        "pineapple", "apricot", "pear", "cantaloupe", "papaya",
        "cherry", "blackberry", "fig", "pomegranate", "tangerine",
        "lemon", "lime", "coconut", "date", "guava",
        "plum", "nectarine", "raspberry", "dragonfruit", "jackfruit"
    ]

    sources = ['192.168.1.10', '10.0.0.5', '172.16.254.1', '203.0.113.45', '198.51.100.75', '127.0.0.1', '192.0.2.88', '203.0.113.25', '10.1.1.1', '192.168.0.255']

    for i in range(1000):
        user_arn = "arn:aws:iam::123456789012:user/JohnDoe"
        table_name = "table_"+ str(i+1)
        source = random.choice(sources)
        timestamps = [datetime(2024, 10, 11, 12, 45, 20) + timedelta(minutes=i * 10) for i in range(4)]

        logs["Records"].append({
            "eventVersion": "1.0",
            "eventTime": timestamps[0].isoformat() + "Z",
            "eventSource": "dynamodb.amazonaws.com",
            "eventName": "CreateTable",
            "userIdentity": {
                "type": "IAMUser",
                "arn": user_arn
            },
            "requestParameters": {
                "tableName": table_name
            },
            "sourceIPAddress": source,
            "userAgent": "aws-cli/2.0.30"
        })

        for _ in range(random.randint(1,4)):
            for _ in range(random.randint(0,5)):
                logs["Records"].append({
                    "eventVersion": "1.0",
                    "eventTime": timestamps[1].isoformat() + "Z",
                    "eventSource": "dynamodb.amazonaws.com",
                    "eventName": "PutItem",
                    "userIdentity": {
                        "type": "IAMUser",
                        "arn": user_arn
                    },
                    "requestParameters": {
                        "tableName": table_name,
                        "item": {
                            "OrderID": {"S": "12345"},
                            "CustomerID": {"S": "98765"}
                        }
                    },
                    "sourceIPAddress": source,
                    "userAgent": "aws-cli/2.0.30"
                })

            for _ in range(random.randint(0,5)):
                logs["Records"].append({
                    "eventVersion": "1.0",
                    "eventTime": timestamps[2].isoformat() + "Z",
                    "eventSource": "dynamodb.amazonaws.com",
                    "eventName": "UpdateItem",
                    "userIdentity": {
                        "type": "IAMUser",
                        "arn": user_arn
                    },
                    "requestParameters": {
                        "tableName": table_name,
                        "key": {
                            "OrderID": {"S": "12345"}
                        },
                        "updateExpression": "SET OrderStatus = :status",
                        "expressionAttributeValues": {
                            ":status": {"S": "SHIPPED"}
                        }
                    },
                    "sourceIPAddress": source,
                    "userAgent": "aws-cli/2.0.30"
                })

        logs["Records"].append({
            "eventVersion": "1.0",
            "eventTime": timestamps[3].isoformat() + "Z",
            "eventSource": "dynamodb.amazonaws.com",
            "eventName": "DeleteTable",
            "userIdentity": {
                "type": "IAMUser",
                "arn": user_arn
            },
            "requestParameters": {
                "tableName": table_name
            },
            "sourceIPAddress": source,
            "userAgent": "aws-cli/2.0.30"
        })

    for i in range(1000, 1020):
        user_arn = "arn:aws:iam::123456789012:user/JohnDoe"
        table_name = "table"+ str(i+1)
        source = random.choice(sources)
        timestamps = [datetime(2024, 10, 11, 12, 45, 20) + timedelta(minutes=i * 10) for i in range(4)]

        logs["Records"].append({
            "eventVersion": "1.0",
            "eventTime": timestamps[0].isoformat() + "Z",
            "eventSource": "dynamodb.amazonaws.com",
            "eventName": "DeleteTable",
            "userIdentity": {
                "type": "IAMUser",
                "arn": user_arn
            },
            "requestParameters": {
                "tableName": table_name
            },
            "sourceIPAddress": source,
            "userAgent": "aws-cli/2.0.30"
        })

        for _ in range(random.randint(1,4)):
            for _ in range(random.randint(0,5)):
                logs["Records"].append({
                    "eventVersion": "1.0",
                    "eventTime": timestamps[1].isoformat() + "Z",
                    "eventSource": "dynamodb.amazonaws.com",
                    "eventName": "PutItem",
                    "userIdentity": {
                        "type": "IAMUser",
                        "arn": user_arn
                    },
                    "requestParameters": {
                        "tableName": table_name,
                        "item": {
                            "OrderID": {"S": "12345"},
                            "CustomerID": {"S": "98765"}
                        }
                    },
                    "sourceIPAddress": source,
                    "userAgent": "aws-cli/2.0.30"
                })

            for _ in range(random.randint(0,5)):
                logs["Records"].append({
                    "eventVersion": "1.0",
                    "eventTime": timestamps[2].isoformat() + "Z",
                    "eventSource": "dynamodb.amazonaws.com",
                    "eventName": "UpdateItem",
                    "userIdentity": {
                        "type": "IAMUser",
                        "arn": user_arn
                    },
                    "requestParameters": {
                        "tableName": table_name,
                        "key": {
                            "OrderID": {"S": "12345"}
                        },
                        "updateExpression": "SET OrderStatus = :status",
                        "expressionAttributeValues": {
                            ":status": {"S": "SHIPPED"}
                        }
                    },
                    "sourceIPAddress": source,
                    "userAgent": "aws-cli/2.0.30"
                })

        logs["Records"].append({
            "eventVersion": "1.0",
            "eventTime": timestamps[3].isoformat() + "Z",
            "eventSource": "dynamodb.amazonaws.com",
            "eventName": "DeleteTable",
            "userIdentity": {
                "type": "IAMUser",
                "arn": user_arn
            },
            "requestParameters": {
                "tableName": table_name
            },
            "sourceIPAddress": source,
            "userAgent": "aws-cli/2.0.30"
        })

    return logs

# Generate logs
non_anomalous_logs = generate_non_anomalous_logs()
anomalous_logs = generate_anomalous_logs()

# Write logs to JSON files
with open('non_anomalous_dynamodb_logs.json', 'w') as non_anomalous_file:
    json.dump(non_anomalous_logs, non_anomalous_file, indent=4)

with open('anomalous_dynamodb_logs.json', 'w') as anomalous_file:
    json.dump(anomalous_logs, anomalous_file, indent=4)

print("Logs generated successfully.")
