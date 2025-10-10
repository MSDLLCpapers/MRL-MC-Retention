import numpy as np
import dedenser
import pandas as pd
import matplotlib.pyplot as plt
#load old point clouds 

pc = np.load('e480c.npy')
rts = pd.read_csv('enamine_480.csv')


sm= ['O=C(CC=1C=CSC1)N2CCOCC32CCOC3',
    'CC=1C=CC=CC1CN2CCCC(O)C2',
    'CC1CCN(CCS1)C(=O)C=2C=C(C)C=CC2F',
    'CN1N=NC=C1C(=O)N2CCCC2C3CCCC3',
    'O=C1NC(=O)C2(CN3CCC2CC3)N1',
    'CC1CC(N(C1)C(=O)CN)C=2C=CC=C(F)C2',
    'CC1(C)CCC(=O)NC1',
    'CC1CCCN(C1)C(=O)C=2C=CC=NC2O',
    'N[C@@H]1C[C@H]1C=2C=CC=CC2C(F)(F)F |&1:1,3,r|',
    'CC1=NC(=NO1)C2CC3CCC(C2)N3',]

sml = []
rl = []
for i in range(len(rts)):
    if rts.Smiles[i] in sm:
        sml.append(i)
    else:
        rl.append(i)


#pc[np.array([i for i in range(len(pc)) if i not in rl])]

rpc_0 = pc[np.array([i for i in range(len(pc)) if i in rl])]#points to keep for first pointcloud

pc[rpc_1]


dd = dedenser.Dedenser(rpc_0, target=.2*1.25,epsilon=.4,min_size = 200,show=True)

out = dd.downsample()

#rl1_index = out
rl1 = np.array(rl)[np.array(out)]#index (with original values) for downsampled pointcloud #1

bl_1 = []#points used by first cloud and test set
for i in rl1:
    bl_1.append(i)

for i in sml:
    bl_1.append(i)


rpc_1 = pc[np.array([i for i in range(len(pc)) if i not in bl_1])]


dd = dedenser.Dedenser(rpc_1, target=.25*1.25,epsilon=.4,min_size = 200,show=True)

out2 = dd.downsample()


rp1 = []#real index available after downsampling once
for i in range(len(rts)):
    if i in bl_1:
        pass
    else:
        rp1.append(i)

rl2 = np.array(rp1)[np.array(out2)]#points for second fold





############
bl_2 = []#points used by first and second cloud and test set
for i in bl_1:
    bl_2.append(i)

for i in rl2:
    bl_2.append(i)
##############

rpc_2 = pc[np.array([i for i in range(len(pc)) if i not in bl_2])]

dd = dedenser.Dedenser(rpc_2, target=.33*1.25,epsilon=.6,min_size=200,show=True)
out3 = dd.downsample()



rp2 = []#real index available after downsampling once
for i in range(len(rts)):
    if i in bl_2:
        pass
    else:
        rp2.append(i)

rl3 = np.array(rp2)[np.array(out3)]

bl_2.sort()

bl_3 = []#points used by first, second, third cloud and test set
for i in bl_2:
    bl_3.append(i)

for i in rl3:
    bl_3.append(i)
#########


rpc_3 = pc[np.array([i for i in range(len(pc)) if i not in bl_3])]


dd = dedenser.Dedenser(rpc_3, target=.50*1.25,epsilon=.6,min_size=100,show=True)
out4 = dd.downsample()

###


rp3 = []#real index available after downsampling once
for i in range(len(rts)):
    if i in bl_3:
        pass
    else:
        rp3.append(i)

rl4 = np.array(rp3)[np.array(out4)]

bl_3.sort()

bl_4 = []#points used by first, second, third cloud and test set
for i in bl_3:
    bl_4.append(i)

for i in rl4:
    bl_4.append(i)




rl5 = np.array([i for i in range(len(pc)) if i not in bl_4])


sml = np.array(sml)

rl1
rl2
rl3
rl4
rl5



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('UMAP_1')
ax.set_ylabel("UMAP_2")
ax.set_zlabel("UMAP_3")

#ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],s=.3,c='k')
ax.scatter(pc[:, 0][rl1], pc[:, 1][rl1], pc[:, 2][rl1],s=2,c='r')
ax.scatter(pc[:, 0][rl2], pc[:, 1][rl2], pc[:, 2][rl2],s=2,c='g')
ax.scatter(pc[:, 0][rl3], pc[:, 1][rl3], pc[:, 2][rl3],s=2,c='orange')
ax.scatter(pc[:, 0][rl3], pc[:, 1][rl3], pc[:, 2][rl3],s=2,c='yellow')
ax.scatter(pc[:, 0][rl5], pc[:, 1][rl5], pc[:, 2][rl5],s=2,c='b')

plt.show()




