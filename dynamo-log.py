import json
import random
from datetime import datetime, timedelta

def generate_log(index):
    # Generate random timestamps
    event_time = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%dT%H:%M:%SZ")
    creation_time = (datetime.now() - timedelta(days=random.randint(30, 60))).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return {
                "eventVersion": "5.05",
                "userIdentity": {
                    "type": "AssumedRole",
                    "principalId": f"AKTTIOSZODNN8SAMPLE:user{index}",
                    "arn": f"arn:aws:sts::155522255533:assumed-role/users/user{index}",
                    "accountId": random.choice(["4344", "353535", "24322234", "6877887"]),
                    "accessKeyId": f"AKTTIOSZODNN8SAMPLE{index}",
                    "sessionContext": {
                        "attributes": {
                            "mfaAuthenticated": random.choice(["true", "false"]),
                            "creationDate": creation_time
                        },
                        "sessionIssuer": {
                            "type": "Role",
                            "principalId": f"AKTTI44ZZ6DHBSAMPLE{index}",
                            "arn": f"arn:aws:iam::499955777666:role/admin-role{index}",
                            "accountId": "499955777666",
                            "userName": f"user{index}"
                        }
                    }
                },
                "eventTime": event_time,
                "eventSource": "dynamodb.amazonaws.com",
                "eventName": random.choice(["DeleteTable", "CreateTable", "PutItem", "UpdateItem", "Query", "Scan"]),
                "awsRegion": random.choice(["us-east-1", "us-east-2", "us-west-1", "us-west-2", "ap-east-1"]),
                "sourceIPAddress": random.choice(["192.3.4.5","29.3.4.4", "45.3.45.5", "24.5.4.3", "45.2.7.8"]),
                "userAgent": "console.aws.amazon.com",
                "requestParameters": {
                    "tableName": random.choice(["Tools", "Users", "Orders", "Products"])
                },
                "responseElements": {
                    "tableDescription": {
                        "tableName": random.choice(["Tools", "Users", "Orders", "Products"]),
                        "itemCount": random.randint(0, 100),
                        "provisionedThroughput": {
                            "writeCapacityUnits": random.randint(5, 100),
                            "numberOfDecreasesToday": random.randint(0, 10),
                            "readCapacityUnits": random.randint(5, 100)
                        },
                        "tableStatus": random.choice(["ACTIVE", "DELETING"]),
                        "tableSizeBytes": random.randint(1000, 100000)
                    }
                },
                "requestID": f"4D89G7D98GF7G8A7DF78FG89AS7GFSO5AEMVJF66Q9ASUAAJG{index}",
                "eventID": f"a954451c-c2fc-4561-8aea-7a30ba1fdf52{index}",
                "eventType": "AwsApiCall",
                "apiVersion": "2013-04-22",
                "recipientAccountId": "155522255533"
            }

# Generate 100 logs and write to file
logs = [generate_log(i) for i in range(200)]
with open("dynamodb_logs.json", "w") as file:
    json.dump(logs, file, indent=4)

print("Generated 100 DynamoDB logs.")
