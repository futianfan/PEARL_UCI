import numpy as np 
### car dataset


def feat_lst_to_feat_dict(feat):
	feat_dict = [{k:lst.index(k) for k in lst} for lst in feat]
	return feat_dict

def file_to_matrix(filename, feat_dict):  #### 
	with open(filename, 'r') as fin:
		lines = fin.readlines()
		lines = [line.rstrip().split(',') for line in lines]
		num_feat = len(lines[0]) - 1
		num_sample = len(lines)
		mat = np.zeros((num_sample, num_feat + 1))
		for i in range(num_feat + 1):
			lst = [line[i] for line in lines]  ### i-th column 
			mat[:,i] = list(map(lambda x:feat_dict[i][x], lst))
		feat = mat[:,:-1]
		label = mat[:,-1]
		return feat, label 
		### numpy array 

def file_to_rulefile(filename, feat_dict, f_feat, f_label):
	fo = open(f_feat, 'w')
	fo2 = open(f_label, 'w')
	with open(filename, 'r') as fin:
		lines = fin.readlines()
		lines = [line.rstrip().split(',') for line in lines]
		num_feat = len(lines[0]) - 1
		num_sample = len(lines)
		for i in range(num_feat):
			### first order
			for k in feat_dict[i]:
				feature_name = '{F_' + str(i) + '=' + str(k) + '}'
				str_all = ['1' if line[i] == k else '0' for line in lines]
				str_all = ' ' + ' '.join(str_all) + '\n'
				fo.write(feature_name + str_all)
			for j in range(i + 1, num_feat):
				for k1 in feat_dict[i]: 
					for k2 in feat_dict[j]:
						feature_name = '{F_' + str(i) + '=' + str(k1)\
						 + ',F_' + str(j) + '=' + str(k2) + '}'
						str_all = ['1' if line[i] == k1 and line[j] == k2 else '0' for line in lines]
						str_all = ' ' + ' '.join(str_all) + '\n'
						fo.write(feature_name + str_all)
		fo.close()
		for k in feat_dict[-1]:
			feature_name = '{label' + '=' + str(k) + '}'
			str_all = ['1' if line[-1] == k else '0' for line in lines]
			str_all = ' ' + ' '.join(str_all) + '\n'
			fo2.write(feature_name + str_all)
		fo2.close()

class VisitDataSeq:
	def __init__(self, line, N = 1865, max_length = 10):
		line = line.split('\t')
		self.label = 1 if line[0] == 'True' else 0
		admis = [int(i) for i in line[2].split()]
		timestamp = [int(i) for i in line[3].split()]
		uniq_time = list(set(timestamp))
		uniq_time.sort()
		idx = [list(filter(lambda x: timestamp[x] == t, range(len(admis)))) for t in uniq_time]
		self.admis = [[admis[j] for j in i] for i in idx]
		self.leng = len(admis)
		self.N = N 
		self.max_length = max_length
		if self.leng > self.max_length:
			self.leng = self.max_length
			self.admis = self.admis[-self.max_length:]

	def lst2mat(self):
		mat = np.zeros((self.max_length, self.N),dtype = np.float32)
		for i in range(self.leng):
			lst = self.admis[i]
			for j in lst:
				mat[i,j] = 1
		return mat, self.leng 

'''
def lines2mat(lines):
	seq_leng = []
	for i in range(len(lines)):
'''

if __name__ == '__main__':
	### part I
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
	fo = './data/rule_feature'
	fo2 = './data/rule_label'
	feature_dict = feat_lst_to_feat_dict(feat)
	feat, lab = file_to_matrix(filename, feature_dict)
	file_to_rulefile(filename, feature_dict, fo, fo2)
	'''

	### visit data seq 
	filename = './data/heart_failure_train_1.txt'
	with open(filename, 'r') as fin:
		lines = fin.readlines()[1:]
		visit = VisitDataSeq(lines[0])















