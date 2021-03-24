# -*- coding:utf-8 -*-

# Deep Streaming Label Learning
# Pepijn Sibbes adapted

from helpers import split_label
import arff
import numpy as np

def load_dataset(dataset, split):
    if dataset == "yeast":
        data_dir = 'datasets/'
        train_data = arff.load(open(data_dir + 'yeast-train.arff', 'rt'))
        train_data = np.array(train_data['data']).astype(np.float)

        test_data = arff.load(open(data_dir + 'yeast-test.arff', 'rt'))
        test_data = np.array(test_data['data']).astype(np.float)

        train_X = train_data[:, :103]
        train_Y_full = train_data[:, 103:]
        train_Y, train_Y_rest = split_label(train_Y_full, split)

        test_X = test_data[:, :103]
        test_Y_full = test_data[:, 103:]
        test_Y, test_Y_rest = split_label(test_Y_full, split)

    if dataset == "nus":
        data_dir = 'datasets/'
        # uncomment if you need to make the npy file
        # train_data = arff.load(open(data_dir + 'nus-wide-full_BoW_l2-train.arff', 'rt'))
        # train_data = np.array(train_data['data'])
        # np.save('nus_train', train_data)
        train_data = np.load('datasets/nus_train.npy')
       

        # test_data = arff.load(open(data_dir + 'nus-wide-full_BoW_l2-test.arff', 'rt'))
        # test_data = np.array(test_data['data'])
        # np.save('nus_test', test_data)
        test_data = np.load('datasets/nus_test.npy')
        train_data, test_data = train_data[:50_000], test_data[:10_000]

         # remove name of image
        train_data = train_data[:,1:].astype(np.float)
        test_data = test_data[:,1:].astype(np.float)

        train_X = train_data[:, :-81]
        train_Y_full = train_data[:, -81:]
        train_Y, train_Y_rest = split_label(train_Y_full, split)

        test_X = test_data[:, :-81]
        test_Y_full = test_data[:, -81:]
        test_Y, test_Y_rest = split_label(test_Y_full, split)
        # exit()
        

    return train_X, train_Y, train_Y_rest, test_X, test_Y, test_Y_rest

