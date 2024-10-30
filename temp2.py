import re

log_file_path = r"\\wsl.localhost\Ubuntu-20.04\var\log\postgresql\postgresql-12-main.log"

eventsMap = {
    "CREATE": 1,
    "INSERT": 2,
    "UPDATE": 3,
    "SELECT": 4,
    "DELETE": 5,
    "ALTER": 6,
    "TRUNCATE": 7,
    "DROP" : 8,
}

patterns = {
    "SELECT": re.compile(r"SELECT .* FROM (\w+)", re.IGNORECASE),
    "INSERT": re.compile(r"INSERT INTO (\w+)", re.IGNORECASE),
    "UPDATE": re.compile(r"UPDATE (\w+)", re.IGNORECASE),
    "DELETE": re.compile(r"DELETE FROM (\w+)", re.IGNORECASE),
    "CREATE": re.compile(r"CREATE TABLE (\w+)", re.IGNORECASE),
    "DROP": re.compile(r"DROP TABLE (\w+)", re.IGNORECASE),
    "ALTER": re.compile(r"ALTER TABLE (\w+)", re.IGNORECASE),
    "TRUNCATE": re.compile(r"TRUNCATE TABLE (\w+)", re.IGNORECASE),
}

def extract_event_table(log_entry):
    for event, pattern in patterns.items():
        match = pattern.search(log_entry)
        if match:
            table_name = match.group(1)
            return {"event": event, "table": table_name}
    return None

parsed_logs = []
with open(log_file_path, "r") as log_file:
    for log_entry in log_file:
        result = extract_event_table(log_entry)
        if result:
            parsed_logs.append(result)

events = {}

for i in parsed_logs:
    if i['table'] not in events:
        events[i['table']] = []
    events[i['table']].append(eventsMap[i['event']])
print(eventsMap)
print(events)
