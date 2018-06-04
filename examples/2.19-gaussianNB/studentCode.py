from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from classify import NBAccuracy

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train,
                          features_test, labels_test)
    return accuracy


#print("why, hello!")
#print("features train:\n{}".format('\n'.join(["OK: {} || train: {} || test: {}".format(str(t == l), str(t), str(l)) for t, l in zip(labels_train, labels_test)])))
#print("labels train:\n{}".format(features_train))
#print("features test:\n{}".format(labels_train))
#print("labels test:\n{}\n".format(features_test))
#print("labels test:\n{}\n".format(labels_test))

print("labels test:\n{}\n".format(features_train))
print("accuracy = {}%".format(submitAccuracy() * 100.0))
