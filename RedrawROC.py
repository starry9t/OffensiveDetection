# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 02:32:43 2018

@author: zy
"""
from sklearn import metrics
import numpy
import MyModel

with open("result.txt",'r') as reader:
    predicted3 = []
    for line in reader.readlines():
        a = float(line.strip())
        p = 1/(1+numpy.exp(-a))
        predicted3.append(p)
predicted3 = numpy.array(predicted3)
l = len(predicted3)
predicted3 = predicted3.reshape(l,1)
numpy.save("predicted3.npy",predicted3)

predicted1 = numpy.load("predicted1.npy")  #LR
n1 = len(predicted1)
predicted1 = predicted1.reshape(n1,1)
predicted2 = numpy.load("predicted2.npy")  #SVMs
n2 = len(predicted2)
predicted2 = predicted2.reshape(n2,1)
expected = numpy.load("expected.npy")
n3 = len(expected)
expected = expected.reshape(n3,1)    

predicted5 = predicted1*0.3+predicted2*0.7 #LR+SVMs
predicted7 = predicted1*0.5+predicted3*0.5  #LR+CNN
predicted8 = predicted2*0.5+predicted3*0.5  #SVMs+CNN
predicted9 = predicted1*0.25+predicted2*0.25+predicted3*0.5

test_l = numpy.load("expected.npy")

print("prediction done.")
auc1 = MyModel.GetMetrics(predicted1,test_l)
print("auc of LR prediction is: ", auc1 )
auc2 = MyModel.GetMetrics(predicted2,test_l)
print("auc of SVMs prediction is: ", auc2 )
auc3 = MyModel.GetMetrics(predicted3,test_l)
print("auc of CNN prediction is: ", auc3 )

auc5 = MyModel.GetMetrics(predicted5,test_l)
print("auc of LR+SVMs prediction is: ", auc5 )
auc7 = MyModel.GetMetrics(predicted7,test_l)
print("auc of LR+CNN prediction is: ", auc7 )
auc8 = MyModel.GetMetrics(predicted8,test_l)
print("auc of SVMs+CNN prediction is: ", auc8 )
auc9 = MyModel.GetMetrics(predicted9,test_l)
print("auc of LR+SVMs+CNN prediction is: ", auc9 )

predicteds = [predicted1,predicted2,predicted3,predicted5,predicted7,predicted8,predicted9]
pic = MyModel.DrawAUC(predicteds,test_l,"ROC curve")
pic.savefig("ROC.png")
pic.show()
