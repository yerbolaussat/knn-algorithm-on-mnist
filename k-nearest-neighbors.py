# Author: Yerbol Aussat

import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import pickle
from sklearn.neighbors import KNeighborsClassifier
import collections
import time
import os.path
import matplotlib.pyplot as plt


# loading training data
def load_training_samples():
	global trn_data
 	trn_data_temp= np.genfromtxt('MNIST_X_train.csv', delimiter=',')
# 	trn_data = trn_data_temp[:500, :]
	trn_data = trn_data_temp
	return trn_data

# loading training labels
def load_training_labels():
	global trn_labels
 	trn_labels_temp = np.genfromtxt('MNIST_y_train.csv', delimiter=',')
# 	trn_labels = trn_labels_temp[:500]
	trn_labels = trn_labels_temp
	return trn_labels

# loading testing data 
def load_testing_samples():
	global test_data
 	test_data_temp = np.genfromtxt('MNIST_X_test.csv', delimiter=',')
# 	test_data = test_data_temp[:500, :]
	test_data = test_data_temp
 	return test_data
		
# loading testing labels
def load_testing_labels():
	global test_labels
 	test_labels_temp = np.genfromtxt('MNIST_y_test.csv', delimiter=',')
# 	test_labels = test_labels_temp[:500]
	test_labels = test_labels_temp
 	return test_labels
 	
# kNN
def kNN(X_train, y_train, X_test, k):
	y_pred = []
	start_time = time.time()

	for t in range(X_test.shape[0]):
		x0 = X_train - X_test[t]
 		D_t = np.sum(x0**2,axis=-1)**(1./2)
 		k_smallest = np.argpartition(D_t, k-1)[:k]  # linear time at the worst case
 		y_k_smallest = y_train[k_smallest]
 		y_prediction = collections.Counter(y_k_smallest).most_common()[0][0]
		y_pred.append(y_prediction)
	print "prediction done in ", time.time() - start_time, "sec"
	return y_pred
	
# error
def calculate_error(test_labels, y_pred):
	return 1 - 1.0 * np.sum(test_labels == y_pred) / len(y_pred)
	
# average validation error
def find_valid_error(X_train, y_train):
	fold_size = np.linspace(0, len(X_train), 11, dtype = int)
	errors = []
	global k_vals
	k_vals = np.linspace(1, 6, 6, dtype = int)
	print k_vals
	
	for k in k_vals:
		
		print '\n', '-'*20
		print 'k', k
		print '-'*20

		errors_for_k = []
		for i in range(10):
			# creating validation sets and reduced training sets that exclude validation set
			X_tr0 = np.delete(X_train, range(fold_size[i], fold_size[i+1]), 0) 
			y_tr0 = np.delete(y_train, range(fold_size[i], fold_size[i+1]), 0) 
			X_valid = X_train[fold_size[i]:fold_size[i+1], :]
			y_valid = y_train[fold_size[i]:fold_size[i+1]]
# 			print X_tr0.shape
# 			print X_valid.shape
			
			y_pred = kNN(X_tr0, y_tr0, X_valid, k)
			error = calculate_error(y_pred, y_valid)
			print "error", i, ":", error
			errors_for_k.append(error)
		error_ave = sum(errors_for_k) / 10
		print "\nerror_ave k", k, ":", error_ave, "\n"
		errors.append(error_ave)
	
	best_k_index = np.argmin(errors)
	best_k = k_vals[best_k_index]
	
#Plotting
# 	plt.plot(k_vals, errors)
# 	plt.plot([best_k], [errors[best_k_index]], marker='o', markersize=6, color="red")
# 	plt.ylabel('Error ')
# 	plt.xlabel('k')
# 	plt.title('Average Validation Error vs k ')
# 	plt.grid(True)
# 	plt.show()
	
	return errors, best_k

def test_on_best_k(X_train, y_train, X_test, y_test, best_k):
	print "calculating test error with k =", best_k
	y_pred = kNN(X_train, y_train, X_test, best_k)
	error = calculate_error(y_test, y_pred)
	print "Test error =", error
	return error

def main():

	if not os.path.isfile('trn_data'):
		trn_data_full = load_training_samples()
		f1=open('trn_data','w')
		print "pickling trn_data..."
		pickle.dump(trn_data_full, f1)	
		print "trn_data pickled"
		f1.close()
  		trn_data = trn_data_full[:20000]
#   	trn_data = trn_data_full
	else:
		f1 = open('trn_data', 'r')
 		trn_data_full = pickle.load(f1)
  		trn_data = trn_data_full[:20000]
#   	trn_data = trn_data_full
		f1.close()
		
	if not os.path.isfile('trn_labels'):
	 	trn_labels_full = load_training_labels()
		f2=open('trn_labels','w')
		print "pickling trn_labels..."
		pickle.dump(trn_labels_full, f2)	
		print "trn_labels pickled"
		f2.close()
		trn_labels = trn_labels_full[:20000]
# 		trn_labels = trn_labels_full
	else:
		f2 = open('trn_labels', 'r')
		trn_labels_full = pickle.load(f2)
		trn_labels = trn_labels_full[:20000]
# 		trn_labels = trn_labels_full
		f2.close()
	
	if not os.path.isfile('test_data'):
	 	test_data = load_testing_samples()
		f3=open('test_data','w')
		print "pickling test_data..."
		pickle.dump(test_data, f3)	
		print "test_data pickled"
		f3.close()
	else:
		f3 = open('test_data', 'r')
		test_data = pickle.load(f3)
	#  	test_data = test_data_temp[:1000]
		f3.close()
	
	if not os.path.isfile('test_labels'):
	 	test_labels = load_testing_labels()
		f4=open('test_labels','w')
		print "pickling test_labels..."
		pickle.dump(test_labels, f4)	
		print "test_labels pickled"
		f4.close()
	else:
		f4 = open('test_labels', 'r')
		test_labels = pickle.load(f4)
	#  	test_labels = test_labels_temp[:1000]
		f4.close()
	
	errors, best_k = find_valid_error(trn_data, trn_labels)
	print "ERRORS:", errors
	print best_k
	
	test_on_best_k(trn_data_full, trn_labels_full, test_data, test_labels, best_k)
		
if __name__ == "__main__":
    main()

