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
        seq.append(1)
        seq += [2] * random.randint(1, 5)
        seq += [3] * random.randint(1, 5)
        seq.append(4)
        sequences.append(seq)
    return sequences

def generate_anomalous_sequences(num_sequences):
    sequences = []
    for _ in range(num_sequences):
        seq = []
        anomaly_type = random.choice(["missing_create", "wrong_order", "premature_delete"])

        if anomaly_type == "missing_create":
            seq += random.choices([2, 3], k=random.randint(1, 4))
            seq.append(4)

        elif anomaly_type == "wrong_order":
            seq += random.choices([3, 4, 2], k=random.randint(1, 4))
            if 1 not in seq:
                seq.append(1)

        elif anomaly_type == "premature_delete":
            seq.append(1)
            seq.append(4)
            seq += random.choices([2, 3], k=random.randint(1, 4))

        sequences.append(seq)
    return sequences