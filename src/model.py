
from __future__ import print_function
import sys
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



class NNParam:
	def __init__(self, input_dim, output_dim, hidden_size = 20, prototype_num = 10, num_of_hidden_layer = 1):
		self.input_dim = input_dim
		self.output_dim = output_dim 
		self.hidden_size = hidden_size 
		self.prototype_num = prototype_num 
		self.num_of_hidden_layer = num_of_hidden_layer

class RNNParam(NNParam):
	def __init__(self, input_dim, output_dim, hidden_size, prototype_num, bidirection = True, \
		maxlength = 10, batch_first = True, rnn_layer = 1, lambda_dictionary = 5e-1):
		NNParam.__init__(self, input_dim, output_dim, hidden_size, prototype_num)
		self.bidirection = bidirection 
		self.maxlength = maxlength
		self.batch_first = batch_first
		self.rnn_layer = rnn_layer
		self.lambda_dictionary = lambda_dictionary

class BaseFCNN(torch.nn.Module):
	def __init__(self, param):
		super(BaseFCNN, self).__init__()
		self.input_dim = param.input_dim 
		self.output_dim = param.output_dim 
		self.hidden_size = param.hidden_size 
		self.num_of_hidden_layer = param.num_of_hidden_layer
		self.fc_in = nn.Linear(self.input_dim, self.hidden_size)
		self.hidden_layer = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for \
			_ in range(self.num_of_hidden_layer)])
		self.fc_out = nn.Linear(self.hidden_size, self.output_dim)
		self.f = F.sigmoid 

	def forward_hidden(self, x):
		for layer in range(self.num_of_hidden_layer):
			x = self.f(self.hidden_layer[layer](x))
		return x

	def forward_a(self, X_batch):
		X_batch = torch.from_numpy(X_batch).float()
		X_batch = Variable(X_batch)
		batch_size = X_batch.shape[0]
		X_hid = self.fc_in(X_batch)
		X_hid = self.forward_hidden(X_hid)
		return X_hid

	def forward(self, X_batch):
		X_hid = self.forward_a(X_batch) 
		X_out = self.fc_out(X_hid)
		return F.softmax(X_out)

	def test(self, X, batch_size = 128):
		N, _ = X.shape 
		num_iter = int(np.ceil(N / batch_size))
		for i in range(num_iter):
			X_batch = X[i * batch_size: i * batch_size + batch_size]
			X_batch_out = self.forward(X_batch)
			X_out = torch.cat([X_out, X_batch_out], 0) if i > 0 else X_batch_out
		return X_out 

class ProtoNN(BaseFCNN, torch.nn.Module):
	def __init__(self, param):
		super(ProtoNN, self).__init__(param)
		self.prototype_num = param.prototype_num
		self.prototype = torch.zeros(self.prototype_num, self.hidden_size)
		self.prototype = Variable(self.prototype)
		self.fc_out = nn.Linear(self.prototype_num, self.output_dim)

	def generate_average_hidden_code(self, data, batch_size = 128):
		N = data.shape[0]
		num_iter = int(np.ceil(N * 1.0 / batch_size))
		for it in range(num_iter):
			stt = it * batch_size 
			endn = min(stt + batch_size, N)
			X_batch_hidden = self.forward_a(data[stt:endn,:])
			X_hidden = X_batch_hidden if it == 0 else torch.cat([X_hidden, X_batch_hidden], 0)
		return torch.mean(X_hidden, 0)

	def generate_prototype(self, data, assignment, batch_size = 128): 
		### [[1,2,6], [0,3,5], [4,7,8]] for example. 
		assert len(assignment) == self.prototype_num
		for i in range(self.prototype_num):
			data_single_prototype = data[assignment[i],:]
			self.prototype[i,:] = self.generate_average_hidden_code(data_single_prototype, batch_size)

	def forward_prototype(self, X_hid):  #### cosine similarity 
		N, d = X_hid.shape 
		p = self.prototype_num
		norm = X_hid.norm(p=2, dim=1, keepdim=True) 
		X_hid = X_hid.div(norm)  ### normalized 
		X_hid = X_hid.view(N,d,1)
		X_hid_ext = X_hid.expand(N,d,p)
		proto_vector = self.prototype.data 
		proto_vector = Variable(proto_vector, requires_grad = False)
		norm = proto_vector.norm(p=2, dim = 1, keepdim = True)
		proto_vector = proto_vector.div(norm)  ### normalized 
		proto_vector = proto_vector.view(1,d,p)
		proto_vector = proto_vector.expand(N,d,p)
		inner_product = X_hid_ext * proto_vector
		inner_product = torch.sum(inner_product, 1)
		inner_product = inner_product.view(N,p)
		return inner_product 

	def forward(self, X_batch):
		X_hid = self.forward_a(X_batch) 
		X_hid = self.forward_prototype(X_hid)
		X_out = self.fc_out(X_hid)
		return F.softmax(X_out)

#### forward dictionary 

class Dictionary:  ## (torch.nn.Module)
	def __init__(self, input_dim, output_dim):
		self.input_dim = input_dim
		self.output_dim = output_dim 
		self.dictionary = Variable(torch.rand(self.input_dim, self.output_dim) / np.sqrt(self.input_dim), requires_grad = True)
		self.lambda_1 = 1e-2
		self.lambda_2 = 1e-1 

	def forward(self, X):
		## X.shape[0] is batch_size
		assert X.shape[1] == self.input_dim 
		X = X.transpose(0,1)
		A = self.dictionary
		AT = A.transpose(0,1)
		ATA = torch.mm(AT,A)
		AAA = torch.inverse(ATA)
		AAAA = torch.mm(AAA, AT)
		AX = torch.mm(AAAA, X)
		loss = torch.norm( X - torch.mm(self.dictionary, AX) )**2
		return AX.transpose(0,1), loss 
	def __str__(self):
		return str(self.dictionary[0,2])

def variable_length_RNN(X, X_len, RNN):
	### X batch_size, leng, dim
	batch_size = len(X_len)
	v2k = sorted(list(range(batch_size)), key = lambda i:X_len[i], reverse = True)
	k2v = {j:i for i,j in enumerate(v2k)}
	k2v = [k2v[i] for i in range(batch_size)]
	X_len_sort = list( np.array(X_len)[v2k] )
	X_sort = X[v2k]
	X_sort_packed = torch.nn.utils.rnn.pack_padded_sequence(X_sort, X_len_sort, batch_first = True)
	_, (X_out, _) = RNN(X_sort_packed, None)
	X_out = torch.cat([X_out[0], X_out[1]], 1)
	X_out = X_out[k2v]
	return X_out


class Dictionary_RNN(torch.nn.Module):
	def __init__(self, param):
		super(Dictionary_RNN, self).__init__()
		self.input_dim = param.input_dim
		self.output_dim = param.output_dim 
		self.hidden_size = param.hidden_size 
		self.prototype_num = param.prototype_num 
		self.bidirection = param.bidirection 
		self.maxlength = param.maxlength 
		self.rnn_layer = param.rnn_layer 
		self.lambda_dictionary = param.lambda_dictionary

		self.dictionary = Variable(torch.rand(self.input_dim, self.prototype_num) / np.sqrt(self.input_dim))
		##!!!##self.dict = Dictionary(self.input_dim, self.prototype_num)

		self.rnn = nn.LSTM(
			input_size = self.prototype_num,
			hidden_size = int(self.hidden_size / 2),
			num_layers = self.rnn_layer,
			batch_first = True, 
			bidirectional = self.bidirection 
			)
		self.out = nn.Linear(self.hidden_size, self.output_dim)

	def dictionary_forward(self, X):
		X = X.transpose(0,1)
		A = self.dictionary
		##print(A[0,1], end = ' ')
		AT = A.transpose(0,1)
		ATA = torch.mm(AT,A)
		AAA = torch.inverse(ATA)
		AAAA = torch.mm(AAA, AT)
		AX = torch.mm(AAAA, X)
		loss = torch.norm( X - torch.mm(self.dictionary, AX) )**2
		return AX.transpose(0,1), loss


	def forward(self, Tens, T_lens):
		assert isinstance(T_lens, list)
		batch_size, maxlength, input_dim = Tens.shape 
		assert maxlength == self.maxlength 
		assert input_dim == self.input_dim
		Tens = Tens.view(-1, input_dim)
		##!!!##Tens2, loss = self.dict.forward(Tens)
		Tens2, loss = self.dictionary_forward(Tens)
		Tens2 = Tens2.view(batch_size, maxlength, -1)
		assert Tens2.shape[2] == self.prototype_num
		X_out = variable_length_RNN(Tens2, T_lens, self.rnn)
		X_out = self.out(X_out)
		return X_out, loss 


