import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score

scaler = sklearn.preprocessing.StandardScaler()

desc = pd.read_csv('MCaRTs_descs.csv')
desc = desc.drop(columns=['Unnamed: 0'])
desc.iloc[:, :-1] = scaler.fit_transform(desc.values[:, :-1])


models = []

column_descs = [[1.13,0.05,.02,-0.05,0.06,0.1,6.5,2.0,2],#C18  
                [1.13,0.05,.02,-0.05,0.06,0.1,6.5,2.0,6.8],#
                [0.50,-0.10,-0.22,0.04,1.04,1.7,1.7,1.8,2],#cyn -0.781
                [0.50,-0.10,-0.22,0.04,1.04,1.7,1.7,1.8,6.8],#
                [.76,-0.07,-0.05,0.06,0.29,0.58,2,1.7,2],#phenyl -.79
                [.76,-0.07,-0.05,0.06,0.29,0.58,2,1.7,6.8],#
                [0.91,-0.01,-0.06,-0.01,0.37,0.63,4.1,1.9,2],#AQ - 0.79
                [0.91,-0.01,-0.06,-0.01,0.37,0.63,4.1,1.9,6.8],
                ]

scaler = sklearn.preprocessing.StandardScaler()

column_descs = np.array(column_descs)
column_descs = scaler.fit_transform(column_descs)

models = [KernelRidge()]

def data_process(labels, train, val):
    val_x = []
    val_y = []
    for exp in val.values:
        s_des = desc.values[desc.index[desc['SMILES']==exp[0]],:-1][0]
        c_des = column_descs[int(labels[exp[1]][0].split('_')[1])]
        rtv = float(labels[exp[1]][0].split('_')[0])
        descs = np.concatenate([np.array(s_des),np.array(c_des)])
        val_x.append(descs)
        val_y.append(rtv)
    train_x = []
    train_y = []
    for exp in train.values:
        s_des = desc.values[desc.index[desc['SMILES']==exp[0]],:-1][0]
        c_des = column_descs[int(labels[exp[1]][0].split('_')[1])]
        rtv = float(labels[exp[1]][0].split('_')[0])
        descs = np.concatenate([np.array(s_des),np.array(c_des)])
        train_x.append(descs)
        train_y.append(rtv)
    return train_x,train_y,val_x,val_y

for model in models:
    for fold in ['F1','F2','F3','F4','F5']:
        labels = pd.read_csv(f'data/{fold}_labels.txt').values
        val = pd.read_csv(f"data/{fold}_val.txt")
        train = pd.read_csv(f"data/{fold}_train.txt")
        train_x,train_y,val_x,val_y = data_process(labels, train, val)
        model.fit(train_x,train_y)
        pred_y = model.predict(val_x)
        r2_score(val_y,pred_y)

    

fold = 'F1'

train_x = []
train_y = []
for exp in train.values:
    s_des = desc.values[desc.index[desc['SMILES']==exp[0]],:-1][0]
    c_des = column_descs[int(labels[exp[1]][0].split('_')[1])]
    rtv = float(labels[exp[1]][0].split('_')[0])
    descs = np.concatenate([np.array(s_des),np.array(c_des)])
    train_x.append(descs)
    train_y.append(rtv)



model = KernelRidge(kernel="rbf", alpha=0.1, gamma=1, C=1e2)
from sklearn.svm import SVR
model = SVR(kernel="poly")
model.fit(train_x,train_y)
pred_y = model.predict(val_x)
r2_score(val_y,pred_y)