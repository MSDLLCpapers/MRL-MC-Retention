import os
import argparse
import random
import copy
import time
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sklearn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import dgl
from dgl.dataloading import GraphDataLoader

from models import GINModel 
from dataset import SMRTDatasetOneHot

from utils import count_parameters, count_no_trainable_parameters, count_trainable_parameters
from dataset import get_node_dim, get_edge_dim
from sklearn.metrics import r2_score

from dgl.nn import SumPooling, AvgPooling


from scipy import stats
from scipy.stats import t
from utils import MC_dropout

column_descs = [[1.13,0.05,.02,-0.05,0.06,0.1,6.5,2.0,2],#C18  
                [1.13,0.05,.02,-0.05,0.06,0.1,6.5,2.0,6.8],#
                [0.50,-0.10,-0.22,0.04,1.04,1.7,1.7,1.8,2],#cyn -0.781
                [0.50,-0.10,-0.22,0.04,1.04,1.7,1.7,1.8,6.8],#
                [.76,-0.07,-0.05,0.06,0.29,0.58,2,1.7,2],#phenyl -.79
                [.76,-0.07,-0.05,0.06,0.29,0.58,2,1.7,6.8],#
                [0.91,-0.01,-0.06,-0.01,0.37,0.63,4.1,1.9,2],#AQ - 0.79
                [0.91,-0.01,-0.06,-0.01,0.37,0.63,4.1,1.9,6.8],
                ]

hyperparam = 0.1

scaler = sklearn.preprocessing.StandardScaler()

column_descs = np.array(column_descs)
column_descs = scaler.fit_transform(column_descs)


true_y = pd.read_csv('data/all_labels.txt').true_y.values

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    dgl.seed(seed)
    dgl.random.seed(seed)

def load_best_model(path2model, model):
    model_state = torch.load(path2model, map_location=device) 
    model.load_state_dict(model_state)  
    print(f"Load model : {path2model}")
    print(f"all params: {count_parameters(model)}\n"
            f"trainable params: {count_trainable_parameters(model)}\n"
            f"freeze params: {count_no_trainable_parameters(model)}\n")
    return model


class ExtendedModel(nn.Module):
    def __init__(self, original_model):
        super(ExtendedModel, self).__init__()
        self.original_model = original_model
        print(self.original_model.forward)
        self.fc = nn.Sequential(
            nn.Linear(500 + 9, 80),
            nn.LayerNorm(80),
            nn.LeakyReLU(),
            nn.Linear(80,50),
            nn.LayerNorm(50),
            nn.PReLU(),
            nn.Linear(50,2)
        )

    def forward(self, bg, descriptors):
        original_output = self.original_model(bg)
        combined = torch.cat((original_output, descriptors), dim=1)
        output = self.fc(combined)
        return output[:,0], output[:,1]


def test(model,device, dataloader, loss_fn, loss_fn_MAE, return_pred=False):
    num_batches = len(dataloader)
    n_forward_pass = 100
    test_loss = 0
    test_loss_MAE = 0
    all_labels = []
    all_preds = []
    model.eval()
    MC_dropout(model)
    test_y_mean = []
    test_y_var = []
    with torch.no_grad():
        for step, (bg, labels) in enumerate(dataloader):
            bg = bg.to(device)
            mean_list, var_list = [], []
            lvs, cn = [],[]
            print(labels)
            for l in labels:
                lvs.append(float(true_y[int(l)].split('_')[0]))
                cn.append(column_descs[int(true_y[int(l)].split('_')[1])])
                
            for l in lvs:
                all_labels.append(float(l))
            
            labels = torch.tensor(lvs,dtype=torch.float32).reshape(-1, 1)
            labels = labels.to(device)
            cn = torch.tensor(np.array(cn),dtype=torch.float32).reshape(-1,9)
            cn = cn.to(device)
            for _ in range(n_forward_pass):
                pred, logvar = model(bg,cn)
                mean = pred
                mean_list.append(mean.cpu().numpy())
                var_list.append(np.exp(logvar.cpu().numpy()))
                for p in pred:
                    all_preds.append(float(p))
                labels = labels.reshape(-1,1)
                mean = mean.reshape(-1,1)
                loss = loss_fn(mean.squeeze(), labels.squeeze())
                loss = (1 - hyperparam) * loss.mean() + hyperparam * (loss * torch.exp(-logvar) + logvar).mean()
                loss = loss.detach().item()
                test_loss += loss
                test_loss_MAE += loss_fn_MAE(labels.squeeze(), pred.squeeze()).item()
                test_loss2 = test_loss_MAE   

            test_y_mean.append(np.array(mean_list).transpose())
            test_y_var.append(np.array(var_list).transpose())
        test_loss_MAE /= n_forward_pass
        test_loss /= n_forward_pass
    ty = []
    test_y_means = test_y_mean
    for v in test_y_mean:
        for v2 in v:
            ty.append(sum(v2)/n_forward_pass)
    ty = np.array(ty).flatten()
    al = np.array(all_labels).flatten()
    tv = []
    for v in test_y_var:
        for v2 in v:
            tv.append(sum(v2)/n_forward_pass) 
    test_y_mean = np.vstack(test_y_mean) * train_y_std + train_y_mean
    test_y_var = np.vstack(test_y_var) * train_y_std ** 2
    
    test_y_pred = np.mean(test_y_mean, 1)
    test_y_epistemic = np.var(test_y_mean, 1)
    test_y_aleatoric = np.mean(test_y_var, 1)
    print(test_y_epistemic)
    print('epistemic')
    for i in test_y_epistemic:
        print(i)
    print('aleatoric')
    for i in test_y_aleatoric:
        print(i)
    print(test_y_aleatoric)
    #np.save('epistemic_all',test_y_epistemic)
    #np.save('aleatoric_all',test_y_aleatoric)
    test_size = len(all_labels)
    # distribution of retention time predictions for each moleculez
    all_pred_flat = np.array(test_y_means).reshape(test_size, -1)

    columns = [f'Prediction_{i}' for i in range(1, n_forward_pass + 1)]
    all_pred = pd.DataFrame(all_pred_flat, columns=columns)
    ap = np.mean(all_pred_flat, axis=1)
    r2 = r2_score(al, ty)

    return (test_loss, test_loss_MAE, al, ty, all_pred,tv) if return_pred else (test_loss, test_loss_MAE, r2)

def main(args):
    seed_torch(1)
    early_stop = args.early_stop
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout
    num_layers = args.num_layers
    epochs = args.epochs
    loss_fn = nn.MSELoss(reduction="none")
    loss_MAE = nn.MSELoss(reduction="mean")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = GINModel(num_node_emb=get_node_dim(),num_edge_emb=get_edge_dim(), num_layers=args.num_layers, emb_dim=args.hid_dim,
                         dropout=args.dropout, gru_out_layer=args.gru_out_layer, JK='concat')
    model.readout = AvgPooling()
    path2model = args.model
    model = load_best_model(path2model, model)
    print("Coding: ", model)
    print(f"model loaded from: {path2model}")
    c = 0
    for param in model.parameters():
        if c < 12:
            param.requires_grad = False
        c += 1
    
    print(f"all params: {count_parameters(model)}\n"
            f"trainable params: {count_trainable_parameters(model)}\n"
            f"freeze params: {count_no_trainable_parameters(model)}\n")

    model.swizzle()
    model = ExtendedModel(model)
    model_state = torch.load('output/uncert/uncert_model.pth', map_location=device) 
    model.load_state_dict(model_state)
    model.to(device)
    raw_dir = "./data"
    test_dataset = SMRTDatasetOneHot(name="all_test", raw_dir=raw_dir)
    test_dataset_2 = SMRTDatasetOneHot(name="all_val", raw_dir=raw_dir)
    
    '''data_loader'''

    test_dataloader = GraphDataLoader(test_dataset, batch_size=len(test_dataset))
    test_dataloader_2 = GraphDataLoader(test_dataset_2, batch_size=len(test_dataset))
    
    all_labels = []
    for step, (bg, labels) in enumerate(test_dataloader):
        bg = bg.to(device)
        lvs, cn = [],[]
        for l in labels:
            lvs.append(float(true_y[int(l)].split('_')[0]))
            for l in lvs:
                all_labels.append(float(l))
    
    global train_y #= np.array(all_labels)
    global train_y_mean #= train_y.mean().item()
    global train_y_std  

    train_y = np.array(all_labels)
    train_y_mean = train_y.mean().item()
    train_y_std = train_y.std().item()
    

    # log_file
    best_loss = float("inf")
    times, log_file = [], []
    best_r2 = float("-inf")
    print("Model testing...")

    test_loss, test_MAE, t_r2 = test(model, device, test_dataloader, loss_fn, loss_MAE)
    test_loss, test_MAE, t_r2 = test(model, device, test_dataloader_2, loss_fn, loss_MAE)



# load_best_model("./basemodels/gin_best_model_weight.pth")
if __name__ == '__main__':
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

    parser = argparse.ArgumentParser(description='GNN_RT_MODEL')
    parser.add_argument('--model', type=str, default='no', help='best model')
    # GNN model args
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers.')
    parser.add_argument('--hid_dim', type=int, default=200, help='Hidden channel size.')
    parser.add_argument('--gru_out_layer', type=int, default=6, help='readout layer')
    parser.add_argument('--norm', type=str, default='layer_norm', help='choose from: batch_norm, layer_norm, none')
    parser.add_argument('--update_func', type=str, default='layer_norm', help='choose from: batch_norm, layer_norm, none')
    # training args
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=80, help='Batch size.')
    parser.add_argument('--early_stop', type=int, default=15, help='Early stop epoch.')
    args = parser.parse_args()
    main(args)
