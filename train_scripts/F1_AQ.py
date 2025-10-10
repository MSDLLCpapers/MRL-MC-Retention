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
#from dataset import TLDataset
from dataset import SMRTDatasetOneHot

from utils import count_parameters, count_no_trainable_parameters, count_trainable_parameters
from dataset import get_node_dim, get_edge_dim
from sklearn.metrics import r2_score

from dgl.nn import SumPooling, AvgPooling


column_descs = [[1.13,0.05,.02,-0.05,0.06,0.1,6.5,2.0,2],#C18 
                [1.13,0.05,.02,-0.05,0.06,0.1,6.5,2.0,6.8],#
                [0.50,-0.10,-0.22,0.04,1.04,1.7,1.7,1.8,2],#cyn 
                [0.50,-0.10,-0.22,0.04,1.04,1.7,1.7,1.8,6.8],#
                [.76,-0.07,-0.05,0.06,0.29,0.58,2,1.7,2],#phenyl
                [.76,-0.07,-0.05,0.06,0.29,0.58,2,1.7,6.8],#
                [0.91,-0.01,-0.06,-0.01,0.37,0.63,4.1,1.9,2],#AQ 
                [0.91,-0.01,-0.06,-0.01,0.37,0.63,4.1,1.9,6.8],
                ]



scaler = sklearn.preprocessing.StandardScaler()

column_descs = np.array(column_descs)
column_descs = scaler.fit_transform(column_descs)


true_y = pd.read_csv('data/F1_labels.txt').true_y.values

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
            nn.LeakyReLU(),
            nn.Linear(50,1)
        )

    def forward(self, bg, descriptors):
        original_output = self.original_model(bg)
        combined = torch.cat((original_output, descriptors), dim=1)
        output = self.fc(combined)
        return output


def train(model, device, dataloader, optimizer, loss_fn, loss_fn_MAE,):
    num_batches = len(dataloader)
    train_loss = 0
    train_loss_MAE = 0
    model.train()
    for step, (bg, labels) in enumerate(dataloader):
        bg = bg.to(device)
        lvs, cn = [],[] 
        for l in labels:
            lvs.append(float(true_y[int(l)].split('_')[0]))
            cn.append(column_descs[int(true_y[int(l)].split('_')[1])])

        labels = torch.tensor(np.array(lvs),dtype=torch.float32).reshape(-1, 1)
        labels = labels.to(device)
        cn = torch.tensor(np.array(cn),dtype=torch.float32).reshape(-1,9)
        cn = cn.to(device)
        pred = model(bg,cn)
        loss = loss_fn(labels, pred)
        train_loss += loss.item()
        # MAE Loss
        loss_MAE = loss_fn_MAE(labels, pred)
        train_loss_MAE += loss_MAE.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / num_batches

def test(model,device, dataloader, loss_fn, loss_fn_MAE, return_pred=False):
    num_batches = len(dataloader)
    test_loss = 0
    test_loss_MAE = 0
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for step, (bg, labels) in enumerate(dataloader):
            bg = bg.to(device)
            lvs, cn = [],[]
            for l in labels:
                lvs.append(float(true_y[int(l)].split('_')[0]))
                cn.append(column_descs[int(true_y[int(l)].split('_')[1])])
                
            for l in lvs:
                all_labels.append(float(l))
            
            labels = torch.tensor(np.array(lvs),dtype=torch.float32).reshape(-1, 1)
            labels = labels.to(device)
            cn = torch.tensor(np.array(cn),dtype=torch.float32).reshape(-1,9)
            cn = cn.to(device)
            pred = model(bg,cn)
            for p in pred:
                all_preds.append(float(p))
            test_loss += loss_fn(labels, pred).item()
            test_loss_MAE += loss_fn_MAE(labels, pred).item()
    
    r2 = r2_score(np.array(all_labels,dtype=float), np.array(all_preds,dtype=float))
    test_loss /= num_batches
    test_loss_MAE /= num_batches
    return (test_loss, test_loss_MAE, labels, pred) if return_pred else (test_loss, test_loss_MAE, r2)

def model_finetune():
    pass

def main(args):
    seed_torch(1)
    early_stop = args.early_stop
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout
    num_layers = args.num_layers
    epochs = args.epochs
    loss_fn = RMSELoss()
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
        print(c)
        if c < 12:
            print(param.data)
            param.requires_grad = False
        c += 1
    model.swizzle()
    model = ExtendedModel(model)
    print(f"all params: {count_parameters(model)}\n"
            f"trainable params: {count_trainable_parameters(model)}\n"
            f"freeze params: {count_no_trainable_parameters(model)}\n")

    model.to(device)
    raw_dir = "./data"
    train_dataset = SMRTDatasetOneHot(name="AQ_F1_train", raw_dir=raw_dir)
    test_dataset = SMRTDatasetOneHot(name="AQ_F1_val", raw_dir=raw_dir)
    big_test_dataset = SMRTDatasetOneHot(name="AQ_F1_test", raw_dir=raw_dir)
    '''data_loader'''
    train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, drop_last = False, shuffle=True)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=len(test_dataset))
    big_test_dataloader = GraphDataLoader(big_test_dataset, batch_size=len(big_test_dataset))
    # log_file
    best_loss = float("inf")
    best_model_stat = copy.deepcopy(model.state_dict())
    times, log_file = [], []
    best_r2 = float("-inf")
    print("Model training...")
    optimizer = optim.Adamax(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=150, verbose=True)
    for i in range(epochs):
        t1 = time.time()
        train_loss = train(model, device, train_dataloader, optimizer, loss_fn, loss_MAE)
        test_loss, test_MAE, t_r2 = test(model, device, test_dataloader, loss_fn, loss_MAE)
        t2 = time.time()
        times.append(t2 - t1)
        print(
            f'Epoch {i} |lr: {optimizer.param_groups[0]["lr"]:.6f} | Train: {train_loss:.4f} | Test: {test_loss:.4f} | Test_r2: {t_r2:.4f}| '
            f'Test_MAE: {test_MAE:.4f} | time/epoch: {sum(times) / len(times):.1f}')
        tt_loss, _, tt_r2 = test(model,device, big_test_dataloader, loss_fn, loss_MAE)
        print("Blind R2: ",tt_r2)
        log_file_loop = [i, optimizer.param_groups[0]["lr"], train_loss, test_loss, tt_loss, tt_r2, test_MAE]
        log_file.append(log_file_loop)
        scheduler.step()
        if test_loss < best_loss:
            es = 0
            best_loss = test_loss
            best_model_stat = copy.deepcopy(model.state_dict())
        else:
            es += 1
            print("Counter {} of {}".format(es, early_stop))
            # early stopping
            if es > early_stop:
                print("Early stop, best_loss: ", best_loss)
                print("Early stop, best_loss: ", np.sqrt(best_loss))
                break
    save_log_file(best_model_stat, log_file, 'output/kfold', 1)
    pred_summary = return_prediction(model, best_model_stat, device, "./output/kfold", loss_fn, loss_MAE, 0, big_test_dataloader, 'test')
    pred_summary = return_prediction(model, best_model_stat, device, "./output/kfold", loss_fn, loss_MAE, 0, test_dataloader, 'val')


def save_log_file(best_model_stat, log_file, file_savepath, fold):
    result = pd.DataFrame(log_file)
    result.columns = ["epoch", "lr", "train_loss",  "test_loss", "tt_loss", "tt_r2", "test_MAE"]
    result.to_csv(os.path.join(file_savepath, f"AQ_F1_log_file.csv"))

    # print min
    index = result.iloc[:,3].idxmin(axis =0)
    print("the index of min loss is shown as follows:",result.iloc[index, :])
    torch.save(best_model_stat, os.path.join(file_savepath, f"AQ_F1_model_weight.pth"))



def return_prediction(model, best_model_stat, device, file_savepath, loss_fn, loss_MAE, fold, dataloader, prefix):
    model.load_state_dict(best_model_stat)
    model.to(device)
    _, _, y, pred = test(model, device, dataloader, loss_fn, loss_MAE, return_pred=True)
    y = y.reshape(-1, 1).cpu()
    pred = pred.reshape(-1, 1).cpu()
    print(f"y_test's shape: {y.shape}; pred's shape: {pred.shape}")
    from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_squared_error

    print('r2: ',r2_score(y, pred))
    # save pred dataset
    result = torch.cat([y, pred], dim=1)
    result = pd.DataFrame(result.cpu().numpy())
    result.columns = ["y_label", "pred"]
    result.to_csv(os.path.join(file_savepath ,f"AQ_F1_{prefix}_results.csv"))
    print(y)
    print(pred)
    rt_summary = {
        "mean_absolute_error": mean_absolute_error(y, pred),
        "median_absolute_error": median_absolute_error(y, pred),
        "r2_score": r2_score(y, pred),
        "mean_squared_error": mean_squared_error(y, pred)
    }
    print(rt_summary)
    return rt_summary



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
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.015, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=80, help='Batch size.')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stop epoch.')
    args = parser.parse_args()
    main(args)
