#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:52:06 2023

@author: higashi
"""

import numpy as np
import sklearn.datasets
import matplotlib.pylab as plt
import sklearn.model_selection
import sklearn.svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from PIL import Image, ImageOps

def class_average(X, y, img_size):
    A = []
    for i in range(10):
        x = X[y == i]
        x = x.mean(0)
        A.append(x.reshape(img_size))
    return A

def cross_validate(clf_type, X, y, cv):
    train_scores = []
    test_scores = []

    if clf_type == 'kNN':
        clf = sklearn.neighbors.KNeighborsClassifier()
    if clf_type == 'MLP':
        clf = sklearn.neural_network.MLPClassifier()
    if clf_type == 'NaiveBayes':
        clf = sklearn.naive_bayes.GaussianNB()
    if clf_type == 'SVM':
        clf = sklearn.svm.SVC()
    scores = sklearn.model_selection.cross_validate(clf, X, y, cv=cv, return_train_score=True)
    
    return scores['train_score'],scores['test_score']

def load_img(test_img_number):
    img = Image.open(str(test_img_number)+'.jpg')
    img_gray = img.convert('L')
    img_invert = ImageOps.invert(img_gray)
    img_resize = img_invert.resize(img_size)
    img_array = np.array(img_resize)/16
    img_reshape = img_array.reshape([-1, np.prod(img_size)])
    return np.array(img_reshape[0])

if __name__=="__main__":
    np.random.seed(0)
    
    #%% 01 visualize averaged image for each classs
    print('\nExecuting 01...')
    data = sklearn.datasets.load_digits()
    X, y = data['data'], data['target']
    feature_names = data['feature_names']
    target_names = data['target_names']
    n_samples, n_features = X.shape
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    img_size = [8, 8]
    ave_imgs = class_average(X, y, img_size) # output matrix (n_samples x 8 x 8)    
    plt.figure(1); plt.clf()
    for cc, class_label in enumerate(class_labels):
        plt.subplot(len(class_labels), 1, cc + 1)
        plt.imshow(ave_imgs[cc], cmap='gray')
        plt.axis('off')
    
    #%% 02 use various classifiers
    print('\nExecuting 02...')
    n_fold = 10
    clf_types = ['kNN', 'MLP', 'NaiveBayes', 'SVM']
    scores = np.zeros([2, len(clf_types)])
    for cc, clf_type in enumerate(clf_types):
        cv_train_scores, cv_test_scores = cross_validate(clf_type, X, y, cv=n_fold)
        print('Performed by {}'.format(clf_type))
        print('\tTrain score:\t{:.2%}'.format(cv_train_scores.mean()))
        print('\tTest score:\t{:.2%}'.format(cv_test_scores.mean()))
        scores[0, cc] = cv_train_scores.mean()
        scores[1, cc] = cv_test_scores.mean()
        
    #%% 03 use your own testing images
    print('\nExecuting 03...')
    clf = sklearn.svm.SVC()
    clf.fit(X, y)
    plt.figure(4); plt.clf()
    for cc, test_img_number in enumerate(class_labels):
        test_x = load_img(test_img_number)
        
        plt.subplot(len(class_labels), 1, cc + 1)
        plt.imshow(test_x.reshape(img_size), cmap='gray')
        plt.axis('off')
        
        y_est = clf.predict(test_x[np.newaxis])
        print('Test class {} image --> classified into class {}.'.format(cc, y_est[0]))
    plt.show()

