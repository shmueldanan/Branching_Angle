'''
Created on 28 Feb 2021

@author: danan
'''
import numpy as np

from collections import namedtuple
MyStruct = namedtuple("MyStruct", "field1 field2 field3")
'''
x=list(range(6))
x.pop(4)
print(x)
'''
#m[0] = MyStruct("foo", "bar", "baz")
x=np.array([[0,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
x2=np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
#print(x[np.where(x!=0)][0])

x_center, y_center = np.argwhere(x==1).sum(0)/np.sum(x)

print(np.any(x))
print(y_center)
print(np.max(np.where(np.logical_or(x==1,x2==1))[0]))

x=[5,3,5,6,7,8]

print(len(x))