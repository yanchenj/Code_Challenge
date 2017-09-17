from scipy.io import wavfile
import glob
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import math
from python_speech_features import mfcc
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
dict = {}
dict_rev = {}

def data_preprocess(directory):
	data,feature,name,label,label_name,freq,length = [],[],[],[],[],[],5000
	# parse input
	for file in glob.glob(directory+"*.wav"):
		freq.append(wavfile.read(file)[0])
		# get the name of each label
		label_name.append(str(file)[13:-7])
		# get the name of each audio file
		name.append(str(file)[13:])

	for file in glob.glob(directory+"*.wav"):
		# read audio data into numpy.array
		data.append(wavfile.read(file)[1][:length])

	# convert labels into number
	for l in label_name:
		if l in dict:
			label.append(dict[l])
		else:
			dict[l] = len(label)
			dict_rev[dict[l]] = l 
			label.append(dict[l])

	# convert list into flaot numpy.array
	data = np.asarray(data)
	data = data.astype(float)
	# normalize data vector	
	for i in range(0,len(feature)):
		data[i] = data[i]/np.linalg.norm(data[i])

	#  generate feature vector
	for i in range(0,len(data)):
		feature.append(mfcc(data[i],freq[i]).reshape(-1))

	feature = np.asarray(feature)
	label = np.asarray(label)
	return [feature, name, label]

def data_preprocess2(directory):
	data,feature,name,freq,length = [],[],[],[],5000

	# parse input
	for file in glob.glob(directory+"*.wav"):
		freq.append(wavfile.read(file)[0])
		# get the name of each audio file
		name.append(str(file)[12:])

	for file in glob.glob(directory+"*.wav"):
		# read audio data into numpy.array
		data.append(wavfile.read(file)[1][:length])

	# convert list into flaot numpy.array
	data = np.asarray(data)
	data = data.astype(float)
	
	# normalize data vector	
	for i in range(0,len(feature)):
		data[i] = data[i]/np.linalg.norm(data[i])

	#  generate feature vector
	for i in range(0,len(data)):
		feature.append(mfcc(data[i],freq[i]).reshape(-1))

	feature = np.asarray(feature)
	return [feature, name]

def train(data, label):
	clf = KNeighborsClassifier(10)
	clf.fit(data,label)
	return clf

def test(clf,data,label=None):
	output = []
	for i in range(0,len(data)):
		pred = clf.predict([data[i]])
		output.append(dict_rev[pred[0]])
	if label != None:
		correct = 0;
		for i in range(0,len(label)):
			print(dict[output[i]],label[i])
			if label[i] == dict[output[i]]:
				correct +=1
		print("accuracy is {} \n".format(float(correct)/len(output)))
	return output

def format_output(name, pred):
	file = open("Chenjun_Yang.csv",'w+')
	for i in range(0,len(pred)):
		res = name[i] + ',' + pred[i] + '\n'
		file.write(res)

if __name__ == '__main__':
	[train_data, train_name, train_label] = data_preprocess("./train_data/")
	[test_data, test_name] = data_preprocess2("./test_data/")

	clf = train(train_data,train_label)
	pred = test(clf,test_data)

	format_output(test_name, pred);
