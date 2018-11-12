
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

import preprocess_car as prepro 
import model 


def feedforward(nnet, feat, label, stt, endn, loss_crossentropy):
	batch_data = feat[stt:endn,:]
	batch_label = label[stt:endn]
	batch_label = torch.from_numpy(batch_label).long()
	batch_label = Variable(batch_label)
	batch_output = nnet(batch_data)
	loss = loss_crossentropy(batch_output, batch_label)
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
		print('Epoch: ' + str(epo) + ': loss is ' + str(loss_in_a_epoch))
	return BaseNN 





if __name__ == '__main__':
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
	print(predicted_label.shape)
	predicted_label = np.argmax(predicted_label, 1)
	predicted_label = list(predicted_label)
	accu = 0
	print(predicted_label)
	for i in range(len(predicted_label)):
		if predicted_label[i] == label[i]:
			accu += 1
	print(accu * 1.0 / len(predicted_label))











