from classifiers import *
from sklearn.ensemble import AdaBoostClassifier

Data = Data()
f=open('DTC.txt','w')
f.write('Single decision tree\n')
for depth in range(1,10):
    r=DTC(Data,depth)
    f.write(str(depth)+' '+str(r.score)+'\n')
f.write('Random forest classifier\n')
for num in range(1,10):
	r= RFC(Data,num)
	f.write(str(num)+' '+str(r.score)+'\n')
f.close()
