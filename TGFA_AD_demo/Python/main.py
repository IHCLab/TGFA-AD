import torch
import os
import numpy as np
import random as rn
import scipy.io as sio
import time 
from Transformer import ASCR_Former
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parameters():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--SEED', type=int, default=1029)
    parser.add_argument('--save_model_path', type=str, default='/home/ihclserver/Desktop/anomaly_model/pth/')
    parser.add_argument('--dataset_path', type=str, default='home/ihclserver/Desktop/anomaly_model/')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--input_size', default=(1,4,100,100), type=int, metavar='N', help='number image width')
    parser.add_argument('--hid_local', default=4,type=int, metavar='N', help='number the local embedding layer hidden layer')
    parser.add_argument('--num_head', default=1, type=int, metavar='N', help='number of attention head')
    parser.add_argument('--depth', default=1, type=int, metavar='N', help='number of attention layer ')
    parser.add_argument('--patch_size', default=(2,2), type=int, metavar='N', help='number of image height')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, metavar='LR',help='initial (base) learnng rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--weight_decay', default=1e-5, type=float, metavar='W',help='weight decay (default: 1e-4)')
    args = parser.parse_args()
    return args

def set_seed(SEED):
    torch.manual_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)

def get_abundance():
    abundance=sio.loadmat('HyperCSI_result.mat')["abundance3D"]
    abundance_tensor=torch.tensor(abundance, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(train_dev)
    return abundance_tensor

def initize_model(args):
    abundance_tensor=get_abundance()
    model=ASCR_Former(args.patch_size,args.hid_local,abundance_tensor).to(train_dev).train()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    return model,optimizer,scheduler,abundance_tensor

def main_loop():
    args=parameters()
    set_seed(args.SEED)
    model,optimizer,scheduler,abundance_tensor=initize_model(args)
    print('-------------- Training begins----------------------------\n')
    start = time.perf_counter()
    for epoch in range(args.epochs):
        loss=model(abundance_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:  
            print('epoch [{}/{}],train:{:.4f}'.format(epoch, args.epochs, loss.item()))
    end = time.perf_counter()
    print('-------------- Training is sucessfully done--------------\n')
    print('training time is :',end-start)
    model.eval()
    with torch.no_grad():
        abundance_tensor=model(abundance_tensor)
        abundance_reconstruction=abundance_tensor.squeeze().permute(1,2,0).cpu().numpy()
        sio.savemat('Cdl.mat', {"Cdl":abundance_reconstruction,"time_training":end-start})
        print('-------save sucess---------') 

if __name__ == "__main__":

     main_loop()
