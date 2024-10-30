import random

events = {
    "E1": "CreateTable",
    "E2": "PutItem",
    "E3": "UpdateItem",
    "E4": "DeleteTable"
}

def generate_non_anomalous_sequences(num_sequences):
    sequences = []
    for _ in range(num_sequences):
        seq = []
        seq += [1] * random.randint(0,1)
        seq += random.choices([2, 3], k=random.randint(0, 4))
        seq += [4] * random.randint(0,1)
        sequences.append(seq)
    return sequences

def generate_anomalous_sequences(num_sequences):
    sequences = []
    for _ in range(num_sequences):
        seq = []
        anomaly_type = random.choice(["premature_delete", "double_create"])

        if anomaly_type == "premature_delete":
            seq += [1] * random.randint(0,1)
            seq += random.choices([2, 3], k=random.randint(0, 4))
            seq.append(4)
            seq += random.choices([2, 3, 4], k=random.randint(1, 4))
        elif anomaly_type == "double_create":
            seq += random.choices([2, 3], k=random.randint(0, 4))
            seq.append(1)
            seq += random.choices([2, 3], k=random.randint(0, 4))
            seq.append(1)
            seq += random.choices([2, 3, 4], k=random.randint(1, 4))
        sequences.append(seq)
    return sequences