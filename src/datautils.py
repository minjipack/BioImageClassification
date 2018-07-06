import csv
import random
import numpy as np



def read_data(name):
    train_list = []
    test_list = []
    blinded_list = []
    assert name in ['easy','moderate', 'difficult']
    if name == "easy":
        with open('EASY_TRAIN.csv') as easy_train_file:
            easy_train_reader = csv.reader(easy_train_file, delimiter=' ')
            for row in easy_train_reader:
                r = ''.join(row)
                r_list = r.split(',')
                instance = [float(element) if i != len(r_list)-1 else element for i, element in enumerate(r_list)]
                train_list.append(instance)

        with open('EASY_TEST.csv') as easy_train_file:
            easy_train_reader = csv.reader(easy_train_file, delimiter=' ')
            for row in easy_train_reader:
                r = ''.join(row)
                r_list = r.split(',')
                instance = [float(element) if i != len(r_list)-1 else element for i, element in enumerate(r_list)]
                test_list.append(instance)

        with open('EASY_BLINDED.csv') as easy_train_file:
            easy_train_reader = csv.reader(easy_train_file, delimiter=' ')
            for row in easy_train_reader:
                r = ''.join(row)
                r_list = r.split(',')
                instance = [float(element) for i, element in enumerate(r_list)]
                blinded_list.append(instance)

    if name == "moderate":
        with open('MODERATE_TRAIN.csv') as easy_train_file:
            easy_train_reader = csv.reader(easy_train_file, delimiter=' ')
            for row in easy_train_reader:
                r = ''.join(row)
                r_list = r.split(',')
                instance = [float(element) if i != len(r_list)-1 else element for i, element in enumerate(r_list)]
                train_list.append(instance)

        with open('MODERATE_TEST.csv') as easy_train_file:
            easy_train_reader = csv.reader(easy_train_file, delimiter=' ')
            for row in easy_train_reader:
                r = ''.join(row)
                r_list = r.split(',')
                instance = [float(element) if i != len(r_list)-1 else element for i, element in enumerate(r_list)]
                test_list.append(instance)

        with open('MODERATE_BLINDED.csv') as easy_train_file:
            easy_train_reader = csv.reader(easy_train_file, delimiter=' ')
            for row in easy_train_reader:
                r = ''.join(row)
                r_list = r.split(',')
                instance = [float(element) for i, element in enumerate(r_list)]
                blinded_list.append(instance)

    if name == "difficult":
        with open('DIFFICULT_TRAIN.csv') as easy_train_file:
            easy_train_reader = csv.reader(easy_train_file, delimiter=' ')
            for row in easy_train_reader:
                r = ''.join(row)
                r_list = r.split(',')
                instance = [float(element) if i != len(r_list)-1 else element for i, element in enumerate(r_list)]
                train_list.append(instance)

        with open('DIFFICULT_TEST.csv') as easy_train_file:
            easy_train_reader = csv.reader(easy_train_file, delimiter=' ')
            for row in easy_train_reader:
                r = ''.join(row)
                r_list = r.split(',')
                instance = [float(element) if i != len(r_list)-1 else element for i, element in enumerate(r_list)]
                test_list.append(instance)

        with open('DIFFICULT_BLINDED.csv') as easy_train_file:
            easy_train_reader = csv.reader(easy_train_file, delimiter=' ')
            for row in easy_train_reader:
                r = ''.join(row)
                r_list = r.split(',')
                instance = [float(element) for i, element in enumerate(r_list)]
                blinded_list.append(instance)
    Xtrain = np.array(train_list)
    Xtest = np.array(test_list)
    Xblinded = np.array(blinded_list)
    return Xtrain, Xtest, Xblinded