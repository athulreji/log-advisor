import sys
sys.path.append('../')
from loglizer.models import IsolationForest

anomaly_ratio = 0.03

def isolationForest(x_train, x_test):
    model = IsolationForest(contamination=anomaly_ratio)
    model.fit(x_train)
    return model.predict(x_test)

