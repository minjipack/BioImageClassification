from datautils import read_data
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import math
import numpy as np
import draw_plots as d
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV


UNCERTAIN_SAMPLES_ITER = 100
INITIAL_SAMPLES_SIZE = 500
TOTAL_BUDGET = 2500
MODE = "difficult"
np.random.seed(1000)
CLF=GaussianNB()
# CLF=SVC(probability=True)
RFECV_ENABLE = False

class UncertaintySampling:
    def __init__(self, data, initial_batch_size, budget,
                 uncertain_instances_size, clf = SVC(probability=True)):
        self.initial_batch_size = initial_batch_size                # initial batch size
        self.uncertain_instances_size = uncertain_instances_size    #
        self.budget = budget-initial_batch_size                     # initialy
        self.le = LabelEncoder()
        self.data = data
        self.le.fit(data[:,-1])
        self.labelled_idxs = np.array(range(self.initial_batch_size))
        self.labelled_idxs_random = np.array(range(self.initial_batch_size))
        self.unlabelled_idxs = np.array(range(self.initial_batch_size,data.shape[0]))
        self.unlabelled_idxs_random = np.array(range(self.initial_batch_size, data.shape[0]))
        self.all_idxs = np.array(range(data.shape[0]))
        self.rfecv = RFECV(LogisticRegression(), cv=2)
        self.clf = clf

    def active_fit(self):
        # Active learning
        # 1. fit
        Xlab = self.data[self.labelled_idxs,:-1].astype(np.float)
        ylab = self.data[self.labelled_idxs,-1]
        Xunlab = self.data[self.unlabelled_idxs,:-1].astype(np.float)
        if RFECV_ENABLE:
            self.clf.fit(self.rfecv.fit_transform(Xlab, self.le.transform(ylab)),
                         self.le.transform(ylab))
        else:
            self.clf.fit(Xlab, self.le.transform(ylab))

        # 2. indices
        if RFECV_ENABLE:
            uncertain_idxs = self.find_highest_entropy_instances(self.clf.predict_proba(self.rfecv.transform(Xunlab)))
        else:
            uncertain_idxs = self.find_highest_entropy_instances(self.clf.predict_proba(Xunlab))

        # 3. delete and append
        self.labelled_idxs = np.append(self.labelled_idxs, uncertain_idxs)
        self.unlabelled_idxs = np.array(list(set(self.unlabelled_idxs) - set(uncertain_idxs)))

        Xlabrand = self.data[self.labelled_idxs_random,:-1].astype(np.float)
        ylabrand = self.data[self.labelled_idxs_random,-1]


        # Random
        # 1. fit
        self.clf.fit(Xlabrand, self.le.transform(ylabrand))

        # 2. indices
        # np.random.shuffle(self.unlabelled_idxs_random)
        random_idxs = self.unlabelled_idxs_random[:self.uncertain_instances_size]

        # 3. delete and append
        self.labelled_idxs_random = np.append(self.labelled_idxs_random, random_idxs)
        self.unlabelled_idxs_random = np.array(list(set(self.unlabelled_idxs_random) - set(random_idxs)))

        # 4. budget decrement
        self.budget -= self.uncertain_instances_size


    def predict(self, test_data):
        Xlab = self.data[self.labelled_idxs, :-1].astype(np.float)
        ylab = self.data[self.labelled_idxs, -1]
        Xtest = test_data.astype(np.float)
        if RFECV_ENABLE:
            self.clf.fit(self.rfecv.fit_transform(Xlab, self.le.transform(ylab)),
                         self.le.transform(ylab))
            return self.le.inverse_transform(self.clf.predict(self.rfecv.transform(Xtest)))
        else:
            self.clf.fit(Xlab, self.le.transform(ylab))
            return self.le.inverse_transform(self.clf.predict(Xtest))

    def predict_random(self, test_data):
        Xlabrand = self.data[self.labelled_idxs_random, :-1].astype(np.float)
        ylabrand = self.data[self.labelled_idxs_random, -1]
        self.clf.fit(Xlabrand, self.le.transform(ylabrand))
        Xtest = test_data.astype(np.float)
        return self.le.inverse_transform(self.clf.predict(Xtest))


    def find_highest_entropy_instances(self, predicted_probabilities):
        entropies = []
        for predicted_probability_distribution in predicted_probabilities:
            entropy = self.calculate_entropy(predicted_probability_distribution)
            entropies.append(entropy)
        uncertain_unlabelled_indices = np.argsort(entropies)[::-1][:self.uncertain_instances_size]
        uncertain_indices = self.unlabelled_idxs[uncertain_unlabelled_indices]
        return uncertain_indices

    def calculate_entropy(self, one_predicted_probability_distribution):
        n = len(one_predicted_probability_distribution)
        sum = 0.0
        for i in range(n):
            sum += -1 * one_predicted_probability_distribution[i] * self.safe_log(one_predicted_probability_distribution[i], n)
        return sum

    def safe_log(self,num, n):
        if num == 0.:
            return 0.
        else:
            return math.log(num, n)


def run_experiment(name):
    data_train, data_test, data_blinded = read_data(name)
    #clf = UncertaintySampling(data_train, INITIAL_SAMPLES_SIZE,  TOTAL_BUDGET, UNCERTAIN_SAMPLES_ITER)
    clf = UncertaintySampling(data_train, INITIAL_SAMPLES_SIZE, TOTAL_BUDGET, UNCERTAIN_SAMPLES_ITER, clf=CLF)


    num_queries = np.array([])
    active_errors_train = np.array([])
    active_errors_test = np.array([])
    active_errors_blinded = np.array([])

    random_errors_train = np.array([])
    random_errors_test = np.array([])
    i = 0

    while clf.budget > 0:
        num_queries = np.append(num_queries, clf.initial_batch_size + i*clf.uncertain_instances_size)
        print "{}th".format(i)

        clf.active_fit()

        # active learner
        train_predictions = clf.predict(data_train[:,:-1])
        test_predictions = clf.predict(data_test[:,:-1])


        # random learner
        train_random_predictions = clf.predict_random(data_train[:, :-1])
        test_random_predictions = clf.predict_random(data_test[:, :-1])
        #blinded_random_predictions = clf.predict_random(data_blinded[:,:].astype(np.float))


        # active learner: train, test
        active_error_train = 1.0-accuracy_score(data_train[:,-1], train_predictions)
        active_error_test = 1.0-accuracy_score(data_test[:,-1], test_predictions)

        # random learner: train, test
        random_error_train = 1.0-accuracy_score(data_train[:,-1], train_random_predictions)
        random_error_test = 1.0-accuracy_score(data_test[:,-1], test_random_predictions)

        print "Active Learner                   | Random Learner     "

        print active_error_train, "                     |", random_error_train
        print active_error_test, "                            |   ", random_error_test

        active_errors_train = np.append(active_errors_train, active_error_train)
        random_errors_train = np.append(random_errors_train, random_error_train)

        active_errors_test = np.append(active_errors_test, active_error_test)
        random_errors_test = np.append(random_errors_test, random_error_test)

        #blinded



        print("=========================================================================")
        i+=1



    # # plot the random and active learner
    # d.error_random_active(num_queries, active_errors_train, random_errors_train, "SVM train()".format(MODE), "SVM_train_{}.png".format(MODE))
    # d.error_random_active(num_queries, active_errors_test, random_errors_test, "SVM test()".format(MODE), "SVM_test_{}.png".format(MODE))
    clf_str = None
    if isinstance(clf.clf,SVC):
        clf_str = "SVC"
    elif isinstance(clf.clf, GaussianNB):
        clf_str = "NB"
    if RFECV_ENABLE:
        rfecv_str = "rfecv"
    else:
        rfecv_str = "norfecv"
    d.error_random_active(num_queries, active_errors_train, random_errors_train, "{} train({},{})".format(clf_str, MODE, rfecv_str),
                          "{}_train_{}_{}.png".format(clf_str, MODE, rfecv_str))
    d.error_random_active(num_queries, active_errors_test, random_errors_test, "{} test({})".format(clf_str, MODE),
                          "{}_test_{}_{}.png".format(clf_str, MODE, rfecv_str))

    idx_list = data_blinded[:,0].astype(np.int)
    matrix_to_save = np.column_stack((idx_list, clf.predict(data_blinded[:,1:].astype(np.float))))

    np.savetxt("{}_BLINDED_PRED_{}_{}.csv".format(MODE, rfecv_str, clf_str), matrix_to_save, delimiter=',', fmt='%s')



if __name__ == '__main__':
    run_experiment(MODE)