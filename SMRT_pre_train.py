import argparse
import copy
import time
import os

import dgl
import numpy as np
import random
import os
import pandas as pd
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dgl.dataloading import GraphDataLoader
from dgl.nn import SumPooling, AvgPooling

from dataset import SMRTDatasetOneHot
from dataset import get_node_dim, get_edge_dim
from utils import count_parameters, count_no_trainable_parameters, count_trainable_parameters
from sklearn.metrics import r2_score
from adabelief_pytorch import AdaBelief

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    dgl.seed(seed)
    dgl.random.seed(seed)


#train loop
def train(model, device, dataloader, optimizer, loss_fn, loss_fn_MAE,):
    num_batches = len(dataloader)
    train_loss = 0
    train_loss_MAE = 0
    model.train()
    for step, (bg, labels) in enumerate(dataloader):

        bg = bg.to(device)
        labels = labels.reshape(-1, 1)
        #labels = (labels-.02)/1.1
        labels = labels.to(device)
        pred = model(bg)
        loss = loss_fn(pred, labels)
        train_loss = loss.item()
        # MAE Loss
        loss_MAE = loss_fn_MAE(pred, labels)
        train_loss_MAE += loss_MAE.item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / num_batches


#test loop
def test(model,device, dataloader, loss_fn, loss_fn_MAE, return_pred=False):
    num_batches = len(dataloader)
    test_loss = 0
    test_loss_MAE = 0
    preds = []
    preds_all_f = []
    labels_all = []
    labels_all_f = []
    model.eval()
    with torch.no_grad():
        for step, (bg, labels) in enumerate(dataloader):
            bg = bg.to(device)
            labels= labels.reshape(-1,1)
            #labels = (labels-.02)/1.1
            for l in labels:
                labels_all_f.append(float(l))
            labels = labels.to(device)
            pred = model(bg)
            for p in pred:
                preds_all_f.append(float(p))
            preds.append(pred)
            labels_all.append(labels)
            test_loss += loss_fn(pred, labels).item()
            test_loss_MAE += loss_fn_MAE(pred, labels).item()
    r2 = r2_score(preds_all_f, labels_all_f)
    test_loss /= num_batches
    test_loss_MAE /= num_batches
    return (test_loss, test_loss_MAE, labels_all, preds) if return_pred else (test_loss, test_loss_MAE, r2)


def main():
    seed_torch(seed=args.seed)
    args.name = 'pre_train_concat'
    #train args
    batch_size = args.batch_size
    early_stop = args.early_stop
    lr = args.lr
    model_name = args.model_name
    loss_fn = nn.SmoothL1Loss(reduction="mean")
    loss_MAE = nn.L1Loss(reduction="mean")
    # check cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #save path
    file_savepath =f"./output/SMRT"
    if not os.path.isdir(file_savepath):
        os.makedirs(file_savepath)
    print(file_savepath)

    '''dataset and data_loader'''

    datapath = 'data'
    train_dataset = SMRTDatasetOneHot(name="SMRT_train", raw_dir=datapath)
    test_dataset = SMRTDatasetOneHot(name="SMRT_test", raw_dir=datapath)
    print("Coding.... Training...")
    train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    print("Coding.... Valid...")
    test_dataloader = GraphDataLoader(test_dataset, batch_size=len(test_dataset))
    print("node dim: ",get_node_dim(args.exclude_node))
    print("edge dim: ",get_edge_dim(args.exclude_edge))

    '''init model'''
    
    if model_name == "GIN":
        from models import  GINModel
        model = GINModel(num_node_emb=get_node_dim(),num_edge_emb=get_edge_dim(), num_layers=args.num_layers, emb_dim=args.hid_dim,
                         dropout=args.dropout, gru_out_layer=args.gru_out_layer,JK='last')
    elif model_name == "GIN_average":
        from models import  GINModel
        model = GINModel(num_node_emb=get_node_dim(),num_edge_emb=get_edge_dim(), num_layers=args.num_layers, emb_dim=args.hid_dim,
                         dropout=args.dropout, gru_out_layer=args.gru_out_layer,JK='last')
        model.readout = AvgPooling()

    model.to(device)
    print('----args----')
    print('\n'.join([f'{k}: {v}' for k, v in vars(args).items()]))
    print('----model----')
    print(model)
    print(f"---------params-------------")
    print(f"all params: {count_parameters(model)}\n"
          f"trainable params: {count_trainable_parameters(model)}\n"
          f"freeze params: {count_no_trainable_parameters(model)}\n")

    #log_file
    best_loss = float("inf")
    best_test_MAE = float("inf")
    best_model = copy.deepcopy(model)
    times, log_file = [], []

    print('---------- Training ----------')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=400, verbose=True)
    # training & validation & testing
    for i in range(args.epochs):
        t1 = time.time()
        train_loss = train(model, device, train_dataloader, optimizer, loss_fn, loss_MAE)
        t2 = time.time()
        times.append(t2 - t1)

        test_loss, test_MAE, test_r2 = test(model, device, test_dataloader, loss_fn, loss_MAE)

        print(f'Epoch {i} |lr: {optimizer.param_groups[0]["lr"]:.6f} | Train: {train_loss:.4f} | Valid: {test_loss:.4f} | '
              f'Valid_MAE: {test_MAE:.4f} | Valid_r2: {test_r2:.4f} | '
              f'time/epoch: {sum(times) / len(times):.1f}')

        #local file log
        log_file_loop = [i, optimizer.param_groups[0]["lr"], train_loss, test_loss, test_MAE, test_r2]
        log_file.append(log_file_loop)
        scheduler.step()

        if test_loss < best_loss:
            es = 0
            best_loss = test_loss
            best_test_MAE = test_MAE
            best_model = copy.deepcopy(model)
        else:
            es += 1
            print("Counter {} of {}".format(es, early_stop))
            # early stopping
            if es > early_stop:
                print("Early stop, best_loss: ", best_loss)
                break


    # save results
    result = pd.DataFrame(log_file)
    result.columns = ["epoch", "lr", "train_loss", "valid_loss", "valid_MAE", "valid_r2"]
    result.to_csv(file_savepath + "/log_file.csv")
    with open(file_savepath+"/namespace.txt", "w") as f:
        f.write(str(vars(args)))
    index = result.iloc[:,3].idxmin(axis =0)
    print("the index of min loss is shown as follows:",result.iloc[index, :])
    torch.save(best_model.state_dict(), file_savepath + "/SMRT_model_weight.pth")
    _, _, y, pred = test(best_model, device, test_dataloader, loss_fn, loss_MAE, return_pred=True)
    y = y[0].reshape(-1, 1).cpu()
    pred = pred[0].reshape(-1, 1).cpu()
    print(y)
    print(type(y))
    result = torch.cat([y, pred], dim=1)
    result = pd.DataFrame(result.cpu().numpy())
    result.columns = ["y_label", "pred"]
    result.to_csv(os.path.join(file_savepath ,f"pred_SMRT.csv"))


if __name__ == '__main__':
    """
    Model and training parameters
    """
    parser = argparse.ArgumentParser(description='GNN_RT_MODEL')
    parser.add_argument('--name', type=str, default="test", help='wandb_running_name')
    parser.add_argument('--dataset', type=str, default='ht', help='Name of dataset.')
    parser.add_argument('--datapath', type=str, default='data', help='Path to dataset')
    parser.add_argument('--model_name', type=str, default='GIN_average', help='Name of model, choose from: GAT, GCN, GIN, AFP, DEEPGNN')

    # GNN model args
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers.')
    parser.add_argument('--hid_dim', type=int, default=200, help='hidden dim.')
    parser.add_argument('--gru_out_layer', type=int, default=6, help='readout layer')
    parser.add_argument('--norm', type=str, default='batch_norm', help='choose from: batch_norm, layer_norm, none')
    parser.add_argument('--update_func', type=str, default='none', help='choose from: batch_norm, layer_norm, none')

    # training args
    parser.add_argument('--epochs', type=int, default=2500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--early_stop', type=int, default=20, help='Early stop epoch.')
    parser.add_argument('--seed', type=int, default=1, help='set seed')

    #inference or not
    parser.add_argument("--inference", action="store_true", help="Whether inference")
    parser.add_argument("--best_ckpt", type=str, help="best_model_ckpt")

    parser.add_argument('--exclude_node', default=None, type=str, help='exclude node')
    parser.add_argument('--exclude_edge', default=None, type=str, help='exclude edge')

    args = parser.parse_args()
    print(args)

    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

    main()

