


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











