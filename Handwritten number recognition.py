import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
import struct
def readfile():
    img_1=open('C:/Users/admin/PycharmProjects/project1/train-images.idx3-ubyte','rb')
    img=img_1.read()
    lbl_1=open('C:/Users/admin/PycharmProjects/project1/train-labels.idx1-ubyte','rb')
    lbl=lbl_1.read()
    return img,lbl
def get_image(img):
    image_index=0
    image_index+=struct.calcsize('>IIII')
    magic,nImage,nImgRows,nImgCols=struct.unpack_from('>IIII',img,0)
    im=[]
    for i in range(1000):
        temp=struct.unpack_from('>784B',img,image_index)
        im.append(np.reshape(temp,(28,28)))
        image_index+=struct.calcsize('>784B')
    return im
def get_label(lbl):
    label_index=0
    label_index+=struct.calcsize('>II')
    return struct.unpack_from('>1000B',lbl,label_index)
if __name__=='__main__':
    image_data,label_data=readfile()
    im=get_image(image_data)
    label=get_label(label_data)

X=[]
Y=[]
y=[]
for i in range(10):
    for k in label:
        if k==i:
            y.append(1)
        else:
            y.append(0)
    Y.append(np.array(y))
    y=[]
for k in range(1000):
    X.append([i/255 for j in im[k] for i in j])
X=np.array(X).T
a=[[X] for i in range(10)]
delta=[[] for i in range(10)]
Delta=[[] for i in range(10)]
sigma=0.001
Theta=[[] for i in range(10)]
for i in range(10):
    Theta[i].append(np.random.rand(20,784)*2*sigma-sigma)
    Theta[i].append(np.random.rand(1,20)*2*sigma-sigma)
    for j in range(200):
        a1=1/(1+np.exp(-Theta[i][0].dot(a[i][0])))
        a[i].append(a1)
        a2=1/(1+np.exp(-Theta[i][1].dot(a1)))
        a[i].append(a2)
        delta2=(a2-Y[i])
        delta1=Theta[i][1].T.dot(delta2)*a1*(1-a1)
        Theta[i][0]=Theta[i][0]-1/1000*delta1.dot(X.T)
        Theta[i][1]=Theta[i][1]*0.999-1/1000*a1.dot(delta2.T)
    np.savetxt(r'C:/Users/admin/PycharmProjects/project1/ex4Layer1Theta_data.txt', Theta[i][0])
    np.savetxt(r'C:/Users/admin/PycharmProjects/project1/ex4Layer2Theta_data.txt', Theta[i][1])




