
import torch
from torch import nn 
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np 
from time import time
from sklearn.metrics import roc_auc_score
torch.manual_seed(1)    # reproducible
np.random.seed(1)

import preprocess as prepro 
import model 


def feedforward(nnet, feat, label, stt, endn, loss_crossentropy):
	batch_data = feat[stt:endn,:]
	batch_label = label[stt:endn]
	batch_label = torch.from_numpy(batch_label).long()
	batch_label = Variable(batch_label)
	batch_output = nnet(batch_data)
	loss = loss_crossentropy(batch_output, batch_label)
	return batch_output, loss 


def feedforward_dict(nnet, train_line, loss_crossentropy):
	batch_size = len(train_line)
	batch_label = [train_line[i].label for i in range(batch_size)]
	batch_label = np.array(batch_label)
	batch_label = torch.from_numpy(batch_label).long() 
	batch_label = Variable(batch_label)
	X_len = []
	for i in range(batch_size):
		mat, ll = train_line[i].lst2mat()
		mat = torch.from_numpy(mat).float()
		X_len.append(ll)
		leng, dim = mat.shape 
		mat = mat.reshape(1, leng, dim)
		all_mat = mat if i == 0 else torch.cat([all_mat, mat], 0)
	all_mat = Variable(all_mat)
	batch_output, loss_dictionary = nnet(all_mat, X_len)
	loss = loss_crossentropy(batch_output, batch_label) ##+ nnet.lambda_dictionary * loss_dictionary
	return batch_output, loss 


def training(feat, label, epoch = 100, batch_size = 128, flg = False ):
	N, d = feat.shape 
	num_class = int(np.max(label) + 1)
	param = model.NNParam(input_dim = d, output_dim = num_class, hidden_size = 10, num_of_hidden_layer = 0)
	if flg:
		BaseNN = model.ProtoNN(param) 
	else:
		BaseNN = model.BaseFCNN(param)  ### BaseFCNN  ProtoNN
	LR = 5e-1
	loss_crossentropy = torch.nn.CrossEntropyLoss()
	opt_  = torch.optim.SGD(BaseNN.parameters(), lr=LR)
	loss_record = []
	iter_in_a_epoch = int(np.ceil(N / batch_size))

	###  random initialization 
	if flg:
		assign = [np.random.randint(BaseNN.prototype_num) for i in range(N)]
		assignment = [list(filter(lambda x:assign[x] == i, list(range(N)))) for i in range(BaseNN.prototype_num)]
	###

	for epo in range(epoch):
		t1 = time()
		loss_in_a_epoch = 0.0 
		for it in range(iter_in_a_epoch):
			if flg:
				BaseNN.generate_prototype(feat, assignment)
			_, loss = feedforward(BaseNN, feat, label, it * batch_size, min(N, it * batch_size + batch_size), loss_crossentropy)
			opt_.zero_grad()
			loss.backward() 
			opt_.step() 
			loss_in_a_epoch += loss.data[0]
		loss_record.append(loss_in_a_epoch)
		#print('Epoch: ' + str(epo) + ': loss is ' + str(loss_in_a_epoch))
		print("Epoch: {}, loss: {}, time: {} secs".format(str(epo), str(loss_in_a_epoch)[:5], str(time() - t1)[:4]))
	return BaseNN 

def test_dictionary(test_data, NN, loss, batch_size):
	leng = len(test_data)
	iter_in_a_epoch = int(np.ceil(leng / batch_size))
	for it in range(iter_in_a_epoch):
		batch_output, _ = feedforward_dict(NN, \
			test_data[it * batch_size: it * batch_size + batch_size],\
			loss)
		output = torch.cat([output, batch_output], 0) if it > 0 else batch_output 
	prediction = []
	label = []
	for i in range(leng):
		prediction.append(float(output[i,1]))
		label.append(test_data[i].label)
	print("accuracy: {}".format(roc_auc_score(label, prediction)))
	return label 

def training_dictionary(train_data, test_data, epoch = 200, batch_size = 16):
	leng = len(train_data)
	input_dim = train_data[0].N 
	param = model.RNNParam(input_dim, 2, hidden_size = 30, prototype_num = 10)
	NN = model.Dictionary_RNN(param)
	LR = 3e-1 
	loss_crossentropy = torch.nn.CrossEntropyLoss()
	opt_  = torch.optim.SGD(NN.parameters(), lr=LR)
	loss_record = []
	iter_in_a_epoch = int(np.ceil(leng / batch_size))
	for epo in range(epoch):
		labe = test_dictionary(test_data, NN, loss_crossentropy, batch_size)
		t1 = time()
		loss_in_a_epoch = 0.0 
		for it in range(iter_in_a_epoch):
			### feed forward
			_, loss = feedforward_dict(NN, \
				train_data[it * batch_size: it * batch_size + batch_size],\
				loss_crossentropy)
			opt_.zero_grad() 
			loss.backward()
			opt_.step()
			loss_in_a_epoch += loss.data[0]
		loss_record.append(loss_in_a_epoch)
		print("Epoch: {}, loss: {}, time: {} secs".format(str(epo + 1), str(loss_in_a_epoch)[7:-2], str(time() - t1)[:4]))
	return NN 


if __name__ == '__main__':
	'''
	feat = [\
 		[ 'low', 'med', 'high', 'vhigh'],\
 		[ 'low', 'med', 'high', 'vhigh'],\
 		['2', '3', '4', '5more'],\
 		['2', '4', 'more'],\
 		['small', 'med', 'big'],\
 		['low', 'med', 'high'],\
 		['unacc','acc','good','vgood']]
	filename = './data/cars/car.data'

	feature_dict = prepro.feat_lst_to_feat_dict(feat)
	feat, label = prepro.file_to_matrix(filename, feature_dict)
	nnet = training(feat, label)
	predicted_label = nnet.test(feat)
	predicted_label = predicted_label.data.numpy() 
	predicted_label = np.argmax(predicted_label, 1)
	predicted_label = list(predicted_label)
	accu = 0
	print(predicted_label)
	for i in range(len(predicted_label)):
		if predicted_label[i] == label[i]:
			accu += 1
	print(accu * 1.0 / len(predicted_label))
	'''

	filename = './data/heart_failure_train_1.txt'
	testfile = './data/heart_failure_test_1.txt'
	with open(filename, 'r') as fin:
		lines = fin.readlines()[1:]
		testlines = open(testfile, 'r').readlines()[1:]
		test_data = list(map(prepro.VisitDataSeq, testlines))
		train_data = list(map(prepro.VisitDataSeq, lines))
		nnet = training_dictionary(train_data, test_data)








