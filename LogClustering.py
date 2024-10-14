import sys
import numpy as np
sys.path.append('../')
from loglizer.models import LogClustering

max_dist = 0.3 # the threshold to stop the clustering process
anomaly_threshold = 0.3 # the threshold for anomaly detection

def logClustrering(x_train, y_train, x_test):
    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    model.fit(x_train[y_train == 0, :]) # Use only normal samples for training
    res = model.predict(x_test)
    nom_res = np.where(res> 0, 1, 0)

    return nom_res

