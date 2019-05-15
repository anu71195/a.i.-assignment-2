import pickle
import operator
import numpy as np


with open ('y_test', 'rb') as f:
	y_test=pickle.load(f)

maxlabel=np.argmax(y_test, axis=1)
for ind in range(8):
	# st='matrix'+str(ind)
	# with open (st, 'rb') as f:
	# 	ans=pickle.load(f)
	# maxind=np.argmax(ans, axis=1)
	# dic={}
	# for i in range(len(maxlabel)):
	#     if maxlabel[i] not in dic:
	#         dic[maxlabel[i]]=[]
	#     if y_test[i][maxlabel[i]]==0:
	#         continue
	#     dic[maxlabel[i]].append(maxind[i])

	st='dictionary' + str(ind)
	with open(st, 'rb') as f:
		dic= pickle.load(f)
	for i in range(7):
		dic2={}
		for x in dic[i]:
			if x not in dic2: dic2[x]=0
			dic2[x]+=1
		k=sorted(dic2.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
		print(k[:5])
	print()
	print()


# with open ('dictionary', 'rb') as f:
# 	dic=pickle.load(f)

# for i in range(7):
# 	dic2={}
# 	for x in dic[i]:
# 		if x not in dic2: dic2[x]=0
# 		dic2[x]+=1
# 	k=sorted(dic2.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
# 	print(k[:5])
