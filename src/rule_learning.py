'''
python src/rule_learing.py --rule_file ./data/corels_rule_list --test_file ./data/cars/car.data 
'''

import sys
import argparse
import os
import numpy as np

def line_to_rule(line):
	if 'then' not in line:
		label = line 
		idx1 = label.find('=')
		idx2 = label.find('}')
		label = label[idx1 +1:idx2]
		return dict(), label 

	idx = line.find('then')
	feat = line[:idx - 1]
	label = line[idx+6:]

	### label 
	idx1 = label.find('=')
	idx2 = label.find('}')
	label = label[idx1+1:idx2]

	### feat 
	idx1 = feat.find('{')
	idx2 = feat.find('}')
	feat = feat[idx1+1:idx2]
	feat = feat.split(',')
	dic = dict()
	for k in feat:
		idx1 = k.find('_')
		idx2 = k.find('=')
		num = k[idx1+1:idx2]
		dic[int(num)] = k[idx2+1:]
	return dic, label 

def file_to_rule(filename):
	lines = open(filename, 'r').readlines() 
	f1 = lambda x:lines[x][:4] == "if ("
	f2 = lambda x:lines[x][:6] == "else ("

	leng = len(lines)
	stt = filter(f1, list(range(leng)))
	endn = filter(f2, list(range(leng)))
	stt = list(stt)[0]
	endn = list(endn)[0]
	rule_list = lines[stt:endn+1]
	rule_list = list(map(line_to_rule, rule_list))
	return rule_list 

def single_rule_single_data(dic, line):
	if len(dic) == 0:
		return True 
	for k,v in dic.items():
		if line[k] != dic[k]:
			return False 
	return True 


def single_sample_predict(line, rule_list):
	for dic, label in rule_list:
		if single_rule_single_data(dic, line):
			return label 



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--test_file", help=" test data file ", type=str)
	parser.add_argument("--rule_file", help="rule file", type=str)
	args = parser.parse_args()
	#rule_file = open(args.rule_file, 'r')

	rule_list = file_to_rule(args.rule_file)

	test_file = open(args.test_file, 'r')
	lines = test_file.readlines() 
	lines = [line.rstrip().split(',') for line in lines]
	accu = 0
	for line in lines:
		predicted_label = single_sample_predict(line, rule_list)
		if predicted_label == line[-1]:
			accu += 1
	print(accu * 1.0 / len(lines))


