# -*- coding: utf-8 -*-
#
# Last modification: 4 July. 2019
# Author: Rayanne Souza

import numpy as np
import matplotlib.pyplot as plt
import itertools   

from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix


# Plot based on https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Normalized confusion matrix'
    else:
        title='Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print('save_confusion_matrix')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('matrix.jpg')
    plt.show()


# Loading model
model = load_model('my_model.h5')

# Loading test set
x_test = np.load('test\x_test.npy')
y_test = np.load('test\y_test.npy')

# Making prediction
y_pred = model.predict(x_test) 
pred_bool = np.argmax(y_pred,axis = 1)   # Convert predictions classes to one hot vectors 
y_test_bool = np.argmax(y_test,axis = 1) # Convert validation observations to one hot vectors
cmatrix = confusion_matrix(y_test_bool, pred_bool) # compute the confusion matrix

# # Evaluating network performance   
plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
plot_confusion_matrix(cmatrix, plot_labels) # plot the confusion matrix
print(classification_report(y_test_bool, pred_bool))

print("Test_accuracy = %f  ;  Test_loss = %f" % (test_acc, test_loss))
print("Val_accuracy = %f  ;  Val_loss = %f" % (val_acc, val_loss))

