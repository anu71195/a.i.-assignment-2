import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

def sigm(x):
	return 1 / (1 + math.exp(-x))

with open('matrix', 'rb') as f:
	x=pickle.load(f)

def hist_node_feat(x,ind):
	x=x[:2704]
	x=x[:,:1369]
	
	for i in range(len(x)):
		for j in range(len(x[0])):
			x[i][j]=sigm(x[i][j])

	len_ = int(np.sqrt(len(x)))
	fig = plt.figure()
	features = np.array([x[k] for k in range(len(x))])
	for x in range(len_):
	    for y in range(len_):
	        ax = fig.add_subplot(len_, len_, len_*y+x+1)
	        ax.matshow(features[len_*y+x].reshape((37,37)), cmap = plt.cm.gray)
	        plt.xticks(np.array([]))
	        plt.yticks(np.array([]))
	st='visnf'+str(ind)+'.jpg'
	plt.savefig(st)


def hist_feat_node(x, ind):
	x=x[:2704]
	x=x[:,:1369]

	for i in range(len(x)):
		for j in range(len(x[0])):
			x[i][j]=sigm(x[i][j])

	x = [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))] 

	len_ = int(np.sqrt(len(x)))
	fig = plt.figure()
	features = np.array([x[k] for k in range(len(x))])
	for x in range(len_):
	    for y in range(len_):
	        ax = fig.add_subplot(len_, len_, len_*y+x+1)
	        ax.matshow(features[len_*y+x].reshape((52,52)), cmap = plt.cm.gray)
	        plt.xticks(np.array([]))
	        plt.yticks(np.array([]))
	st='visfn'+str(ind)+'.jpg'
	plt.savefig(st)
