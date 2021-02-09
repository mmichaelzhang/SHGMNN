import matlab.engine
import math
def cal_inference():
	alpha = [0.5, 0.5]
	name = '../Data/'
	train_idx = []
	with open(name + 'Train_idx.txt','r') as f:
	    for line in f:
	        train_idx.append(int(line.strip('\n')))

	valid_idx = []
	with open(name + 'Validate_idx.txt','r') as f:
	    for line in f:
	        valid_idx.append(int(line.strip('\n')))

	test_idx = []
	with open(name + 'Test_idx.txt','r') as f:
	    for line in f:
	        test_idx.append(int(line.strip('\n')))

	edge_index1 = [[],[]]
	edge_attr1 = []
	with open(name + 'edges1.txt','r') as f:
	    for line in f:
	        a = int(line.strip('\n').split('\t')[0])
	        b = int(line.strip('\n').split('\t')[1])
	        c = float(line.strip('\n').split('\t')[2])
	        edge_index1[0].append(a)
	        edge_index1[1].append(b)
	        edge_attr1.append([c])

	edge_index2 = [[],[]]
	edge_attr2 = []
	with open(name + 'edges2.txt','r') as f:
	    for line in f:
	        a = int(line.strip('\n').split('\t')[0])
	        b = int(line.strip('\n').split('\t')[1])
	        c = float(line.strip('\n').split('\t')[2])
	        edge_index2[0].append(a)
	        edge_index2[1].append(b)
	        edge_attr2.append([c])

	y = []
	with open(name + 'Labels.txt','r') as f:
	    for line in f:
	        y.append([int(line.strip('\n'))])

	import numpy as np 
	A = np.zeros([len(y), len(y)])
	for i in range(len(edge_index1[0])):
		A[edge_index1[0][i], edge_index1[1][i]] += alpha[0] * edge_attr1[i][0]
	for i in range(len(edge_index2[0])):
		A[edge_index2[0][i], edge_index2[1][i]] += alpha[1] * edge_attr2[i][0]

	non_train_idx = {}
	count = 0
	for e in valid_idx:
		non_train_idx[e] = count
		count += 1
	for e in test_idx:
		non_train_idx[e] = count
		count += 1

	f1 = open(name + 'A.dat','w')
	f2 = open(name + 'b.dat','w')
	print("Generating input for ADMM..\n")
	count = 1
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			if (A[i,j] != 0 and ((i not in train_idx) or (j not in train_idx))):
				b = 0
				w = A[i,j]
				if (i in train_idx):
					b = y[i][0] * w
					f1.write(str(count) + '\t' + str(non_train_idx[j]+1) + '\t' + str(w) + '\n')
					f2.write(str(b) + '\n')
					count += 1
				elif (j in train_idx):
					b = y[j][0] * w
					f1.write(str(count) + '\t' + str(non_train_idx[i]+1) + '\t' + str(w) + '\n')
					f2.write(str(b) + '\n')
					count += 1
				else:
					f1.write(str(count) + '\t' + str(non_train_idx[i]+1) + '\t' + str(w) + '\n')
					
					f1.write(str(count) + '\t' + str(non_train_idx[j]+1) + '\t' + str(-w) + '\n')
					f2.write(str(0) + '\n')
					count += 1

	with open(name + 'non_train_idx.txt','w') as f:
		for i,x in enumerate(non_train_idx):
			f.write(str(x) + '\t' + str(non_train_idx[x]) + '\n')

