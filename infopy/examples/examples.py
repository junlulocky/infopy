import numpy as np
import sys

sys.path.append("..")

import infopy.infopy as ipy


labels_true = np.array([1,2,3,4])
labels_pred = np.array([0,1,2,2])
print ipy.mutual_information(labels_true, labels_true)
print ipy.mutual_information(labels_pred, labels_pred)
print ipy.mutual_information(labels_true, labels_pred)


print ipy.information_variation(labels_true, labels_pred)