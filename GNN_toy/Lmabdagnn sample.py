#lines 113-180 is decent sample for making graphss
#GNN starts on 220

import numpy as np 
import awkward as ak 
import pandas as pd 
#import dask
#import dask.dataframe as dd

# I/O 
import uproot as ur


# Plotting
import matplotlib.pyplot as plt 

# Physics
from particle import PDGID
 
# Miscellaneous 
import os
import sys #NOTE: ADDED
import tqdm

# ML Imports
import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader, random_split
import torch_geometric as tg
# HIPO bank reading and linking functions

from torch_geometric.data import Data as dt

from torch_geometric.data import Dataset, download_url, InMemoryDataset

 #third is list of graphss   
third = []
kkkk = 0
#for j in range(1):    
 #   number = 5
    #if j==2:
      #  number = 2
j=0  
number = 5
#for j in range(3):
 #   if j==3:
 #       number = 2
for ii in range(number): 
    if j==0:
        filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3313_'+str(ii)+'.hipo'
    if j==1:
        filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3316_'+str(ii)+'.hipo'
    if j==2:
        filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3317_'+str(ii)+'.hipo'
    if j==3 :
        filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3318_'+str(ii)+'.hipo'
    if j==4:
        filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3327_'+str(ii)+'.hipo'

    step = 40000
    for batch in hp.iterate([filee],banks=["MC::Lund", "REC::Particle", "REC::Traj", "REC::Kinematics"],step=step): 
       # print(batch.keys()) 
        #batch.keys()
        dic = batch.keys() 
        break   

    #for battch in hp.iterate(['/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3313_0.hipo'],banks=["REC::Kinematics"],step=step):
        #print(battch.keys()) 
        #batch.keys()
        #dic = battch.keys() 
       # break  
    #
    all_keys            = list(batch.keys())
    rec_particle_name   = 'REC::Particle'
    rec_particle_keys   = get_bank_keys(rec_particle_name,all_keys)
    rec_traj_name       = 'REC::Traj'
    rec_traj_keys       = get_bank_keys(rec_traj_name,all_keys)
    rec_kinematics_name = 'REC::Kinematics' 
    rec_kinematics_keys = get_bank_keys(rec_kinematics_name,all_keys)
    mc_lund_name        = 'MC::Lund'
    mc_lund_keys        = get_bank_keys(mc_lund_name,all_keys) 

    # rec_kinematics_keys = ['REC::Kinematics_idxe', 'REC::Kinematics_idxp', 'REC::Kinematics_idxpi', 'REC::Kinematics_Q2', 
    #                       'REC::Kinematics_nu', 'REC::Kinematics_W', 'REC::Kinematics_x', 'REC::Kinematics_y', 'REC::Kinematics_z', 'REC::Kinematics_xF', 'REC::Kinematics_mass']
    #fulll = [] 
    tst = []
    y=[]


    #j is proton, kk is pion


    kinn = []  
    idkkk =[]
    for i in range(step):
        # Get REC::Particle bank
        rec_particle_event_table = get_event_table(rec_particle_keys,i,batch,dtype=float)

        # Get REC::Traj bank
        rec_traj_event_table = get_event_table(rec_traj_keys,i,batch,dtype=float)

        # Get REC::Kinematics bank
        rec_kinematics_event_table = get_event_table(rec_kinematics_keys,i,batch,dtype=float)

        # Get MC::Lund bank and MC->REC matching indices
        mc_lund_event_table = get_event_table(mc_lund_keys,i,batch,dtype=float)
        match_indices  = get_match_indices(rec_particle_event_table,mc_lund_event_table)

        decay = [3122,[2212,-211]] 
        has_decay, rec_indices = check_has_decay(rec_particle_event_table,mc_lund_event_table,match_indices,decay,
                                       rec_particle_pid_idx=0,mc_lund_pid_idx=3,mc_lund_parent_idx=4,mc_lund_daughter_idx=5)

        #print(rec_indices)
        #getting the index of where the electron proton and pion r 
        savee = batch['REC::Particle_pid'][i]
        j = savee.index(2212); kk = savee.index(-211); elc = savee.index(11) #elc is electron, used for checking that it matches up, j is proton, kk is pionn 
        #ckeck
        #making sure that it is not bad (for the match) #if is bad then lots more stuff to do 
        if match_indices[kk][1]!=-1: 
            #if (np.sqrt(batch['MC::Lund_px'][i][match_indices[kk][1]]**2+batch['MC::Lund_py'][i][match_indices[kk][1]]**2+batch['MC::Lund_pz'][i][match_indices[kk][1]]**2)<1):
            if kk in rec_indices and has_decay == True:    
                track_hit = []
                indexp = [index for (index, item) in enumerate(batch['REC::Traj_pindex'][i]) if item == j]
                indexpi = [index for (index, item) in enumerate(batch['REC::Traj_pindex'][i]) if item == kk]
                #get all the data we need here. 

                totalpos =[ [batch['REC::Traj_x'][i][ii] for ii in indexpi], [ batch['REC::Traj_y'][i][ii] for ii in indexpi], [ batch['REC::Traj_z'][i][ii] for ii in indexpi],
                          [batch['REC::Traj_cx'][i][ii] for ii in indexpi], [ batch['REC::Traj_cy'][i][ii] for ii in indexpi], [ batch['REC::Traj_cz'][i][ii] for ii in indexpi]]


                s = 0
                ss =0 
                for dd in range(len(indexpi)):
                    track_hit += [[batch['REC::Traj_x'][i][indexpi[dd]], batch['REC::Traj_y'][i][indexpi[dd]], batch['REC::Traj_z'][i][indexpi[dd]],
                                 batch['REC::Traj_cx'][i][indexpi[dd]], batch['REC::Traj_cy'][i][indexpi[dd]], batch['REC::Traj_cz'][i][indexpi[dd]],
                                   batch['REC::Particle_px'][i][kk], batch['REC::Particle_py'][i][kk], batch['REC::Particle_pz'][i][kk]]]
                                 # batch['REC::Particle_vx'][i][kk], batch['REC::Particle_vy'][i][kk], batch['REC::Particle_vz'][i][kk],
                                   #random.gauss(batch['REC::Particle_px'][i][kk], 1), random.gauss(batch['REC::Particle_py'][i][kk], 1), 
                                   #random.gauss(batch['REC::Particle_pz'][i][kk], 1),
                                  # random.gauss(batch['REC::Particle_vx'][i][kk], 3.394), random.gauss(batch['REC::Particle_vy'][i][kk], 3.394), 
                                  # random.gauss(batch['REC::Particle_vz'][i][kk], 3.394)]]
                    s +=1
                #track_hit+=[[batch['REC::Particle_px'][i][kk], batch['REC::Particle_py'][i][kk], batch['REC::Particle_pz'][i][kk],
                 #           batch['REC::Particle_vx'][i][kk], batch['REC::Particle_vy'][i][kk], batch['REC::Particle_vz'][i][kk]]]
                ss =0 
                for dd in range(len(indexp)): 
                    track_hit += [[batch['REC::Traj_x'][i][indexp[dd]], batch['REC::Traj_y'][i][indexp[dd]], batch['REC::Traj_z'][i][indexp[dd]],
                                   batch['REC::Traj_cx'][i][indexp[dd]], batch['REC::Traj_cy'][i][indexp[dd]], batch['REC::Traj_cz'][i][indexp[dd]],
                                  batch['REC::Particle_px'][i][j], batch['REC::Particle_py'][i][j], batch['REC::Particle_pz'][i][j]]]
                                  #batch['REC::Particle_vx'][i][j], batch['REC::Particle_vy'][i][j], batch['REC::Particle_vz'][i][j],
                                   #random.gauss(batch['REC::Particle_px'][i][j], 1), random.gauss(batch['REC::Particle_py'][i][j], 1), 
                                   #random.gauss(batch['REC::Particle_pz'][i][j], 1),
                                  # random.gauss(batch['REC::Particle_vx'][i][j], 3.394), random.gauss(batch['REC::Particle_vy'][i][j], 3.394), 
                                  # random.gauss(batch['REC::Particle_vz'][i][j], 3.394)]]

                    ss+=1  


                y = [batch['MC::Lund_vz'][i][match_indices[kk][1]] ] 
                #y = [np.sqrt(batch['MC::Lund_px'][i][match_indices[kk][1]]**2+batch['MC::Lund_py'][i][match_indices[kk][1]]**2+batch['MC::Lund_pz'][i][match_indices[kk][1]]**2)]  

                #track_hit += [[np.sqrt(batch['REC::Particle_px'][i][kk]**2+batch['REC::Particle_py'][i][kk]**2+ batch['REC::Particle_pz'][i][kk]**2)]]

                start = []
                end = []
                for gg in range(s-1):
                    start.append(gg)
                    end.append(gg+1)
                for ggg in range(ss-1): 
                    start.append(s+ggg) 
                    end.append(s+ggg+1)
                #fourth = dt(x=torch.tensor(track_hit), edge_index = torch.tensor([start, end]), y = torch.tensor(y))
                fourth = dt(x=torch.tensor(track_hit), edge_index = torch.tensor([start, end]), y = torch.tensor(y))
               # fourth.kinematics = [battch['REC::Kinematics_Q2'][i][0], battch['REC::Kinematics_nu'][i][0], battch['REC::Kinematics_W'][i][0], battch['REC::Kinematics_x'][i][0],  
                #                     battch['REC::Kinematics_y'][i][0], battch['REC::Kinematics_z'][i][0], battch['REC::Kinematics_xF'][i][0], battch['REC::Kinematics_mass'][i][0]]
                fourth.rec  = [batch['REC::Particle_vz'][i][kk]] 
                #fourth.mom = [np.sqrt(batch['REC::Particle_px'][i][kk]**2+batch['REC::Particle_py'][i][kk]**2+batch['REC::Particle_pz'][i][kk]**2)]
                fourth.elc = [batch['REC::Particle_vz'][i][0] ] 
                fourth.momentum = [batch['MC::Lund_pz'][i][match_indices[kk][1]]]
                #fourth.rec = [np.sqrt(batch['REC::Particle_px'][i][kk]**2+batch['REC::Particle_py'][i][kk]**2+batch['REC::Particle_pz'][i][kk]**2)]
                #idkkk+=[[batch['REC::Particle_vz'][i][kk], batch['REC::Particle_vz'][i][j]]]
                #invarient masss#######################################################################################################
                fourth.imass = [batch['REC::Kinematics_mass'][i][0]] 
                if kk in rec_indices and has_decay == True:
                    fourth.LamT = [1]
                else :
                    fourth.LamT = [0]
                third+= [fourth]
    kkkk+=1
    print(kkkk) 
ii=2 
if j==0:
    filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3313_'+str(ii)+'.hipo'
if j==1:
    filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3316_'+str(ii)+'.hipo'
if j==2:
    filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3317_'+str(ii)+'.hipo'
if j==3 :
    filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3318_'+str(ii)+'.hipo'
if j==4:
    filee = '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_2_12_24/analysis2/kinematics_out_skim_50nA_OB_job_3327_'+str(ii)+'.hipo'

data_list = third  
root = '/work/clas12/users/mencke/dikkk'           
class MyDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None): 
        self.data_list = data_list
        super().__init__(root, transform)  
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self): 
        return 'data.pt' 
 
    def process(self):
        self.save(self.data_list, self.processed_paths[0])      

MyDataset(root, third)

#Now network

import os    
import numpy as np
import numpy.ma as ma   
#import awkward as ak 
from tqdm import tqdm 
import torch 
import torch_geometric as tg  
import torch_geometric 
from torch_geometric.data import Data 
#import torch 
from torch_geometric.data import InMemoryDataset, download_url  
import torch_geometric.transforms as T 

#NOTE: NEW 2/20/23      
from typing import List, Union     

from torch_geometric.data import Data, HeteroData 
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform       
torch.cuda.empty_cache()  
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.norm import GraphNorm, BatchNorm 
from torch.utils.data import random_split 
from torch_geometric.loader import DataLoader 

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = None

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0]) 

#root = '/hpc/group/vossenlab/kam264/L_imass3'
#root = '/hpc/group/vossenlab/kam264/onlylambda_25k' 
#root = '/hpc/group/vossenlab/kam264/Lambda_13_16'
#root = '/hpc/group/vossenlab/kam264/Lambda_1all' 

batch_size = 16 
LR =1e-2
torch.cuda.empty_cache()   

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels): 
        super(GCN, self).__init__()
#         torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels)#.jittable() #NOTE: NEEDED FOR DEPLOYMENT IN CMAKE
        self.conv2 = GCNConv(hidden_channels, hidden_channels)#.jittable()
        #self.block2 = nn.DataParallel(self.block2)
        #self.conv2 = torch.nn.DataParallel(self.conv2) #this was trying the parallization thing. 
        self.conv3 = GCNConv(hidden_channels, hidden_channels)#.jittable()
        #self.conv3 = torch.nn.DataParallel(self.conv3)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)
        self.bn1 = torch_geometric.nn.norm.GraphNorm(hidden_channels)
        self.bn2 = torch_geometric.nn.norm.GraphNorm(hidden_channels)
        self.bn3 = torch_geometric.nn.norm.GraphNorm(hidden_channels)

    def forward(self, x, edge_index, batch):  
        
        x = self.conv1(x, edge_index) #input layer                             
                                                      
        x = self.bn1(x) #normalize it                                          

        x = x.relu() #activation                                               
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
#         print("x.relu() = ",x)  
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
#         # 2. Readout layer                                                   
        x = global_mean_pool(x, batch)
        # 3. Apply a final classifier                                          
        x = F.dropout(x, p=0.5, training=self.training) #for overfittin        
        x = self.lin3(x) 
 
        return x
def RMSELoss(out,y):
    return torch.sqrt(torch.mean((out-y)**2))
def train():
    model.train() #initailize the model                                                                                                                                                                                                        
    #for i, data in tqdm(enumerate(train_loader)): #perhaps tqdm(enumerate(train_loader)), i is index, data jsut moves through all the dtaa in trainingg                                                                                      
    for i,data in enumerate(train_loader):
        data = data.to(device) #switch to GPU                                                                                                                                                                                                 
        optimizer.zero_grad() #                                                                                                                                                                                                               
        out = model(data.x, data.edge_index, data.batch).to(device)  # Perform a single forward pass                                                                                                                                          
        yy = []
        for j in range(0,len(out)):
            yy+= [[data.y[j].item()]]

        yy = torch.tensor(yy).to(device) 
        #print(out)
        #print(yy)
        loss = losss(out, yy).to(device) #compute the loss  
        #print(loss)
        loss.backward() #get the gradients.                                                                                                                                                                                                   
        optimizer.step() 
def test(loader): 
    length = len(loader.dataset)
    model.eval() #evaluate teh model.                                                                                                                                                                                                         

    #mse_tot = []                                                                                                                                                                                                                             
    mse_total = 0
    mse_pi = 0
    mse_p = 0
    #r                                                                                                                                                                                                                                        
    #for data in tqdm(loader):  # Iterate in batches over the training/test dataset.                                                                                                                                                          
    for data in loader:
        data = data.to(device) #put to GPU                                                                                                                                                                                                    
        out = model(data.x, data.edge_index, data.batch).to(device) #evalueate                                                                                                                                                                
        #this and the for loop is converting data.y to a tensor in the same shape as out rows and 2 columns first is y_pion second is y_proton                                                                                                
        yy = []
        for j in range(0,len(out)):
            yy+= [[data.y[j].item()]]
        yy = torch.tensor(yy).to(device) 
        loss = losss(out, yy).cpu() #getting teh loss function                                                                                                                                                                                
        mse_total+=loss.item() #getting the mse (total)                                                                                                                                                                                       

    return mse_total/length 
def print_out():
    model.eval() #initailize the model                                                                                                                                                                                                        
    #for i, data in tqdm(enumerate(train_loader)): #perhaps tqdm(enumerate(train_loader)), i is index, data jsut moves through all the dtaa in trainingg                                                                                      
    outt= []
    for i,data in enumerate(test_loader):
        data = data.to(device) #switch to GPU                                                                                                                                                                                                 
        optimizer.zero_grad() #                                                                                                                                                                                                               
        out = model(data.x, data.edge_index, data.batch).to(device)  # Perform a single forward pass                                                                                                                                          
        out = out.cpu()

        outt+=[[out.detach().numpy()]]
    return outt

def print_outb():
    model.eval() #initailize the model                                                                                                                                                                                                        
    #for i, data in tqdm(enumerate(train_loader)): #perhaps tqdm(enumerate(train_loader)), i is index, data jsut moves through all the dtaa in trainingg      

    outt= [] 
    #for i,data in enumerate(dataset):
    for i,data in enumerate(test_loader):
        data = data.to(device) #switch to GPU                                                                                                                                                                                                 
        optimizer.zero_grad() #                                                                                                                                                                                                               
        out = model(data.x, data.edge_index, data.batch).to(device)  # Perform a single forward pass                                                                                                                                          
        out = out.cpu()

        neww = []
        for j in range(len(data)):
            neww.append(out[j].item() -data.elc[j][0]) 
        outt+=[[neww]]
    return outt  
#model = GCN(dataset.num_node_features,64,2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
#devicee = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') 
print("Device = ",device) 

print("DEBUGGING: torch.cuda.is_available() = ",torch.cuda.is_available())
train_metrics = {'mse':[], "rmse":[] } 
vall_metrics = {'mse':[], "rmse":[] }

for iii in range(1):    
    if iii>0: 
        #del dataset
        del train_loader  
        del val_loader 
    if iii==0:
        root = '/hpc/group/vossenlab/kam264/Qmom0'
    if iii==1:
        root = '/hpc/group/vossenlab/kam264/Qmom1'
    if iii==2:
        root = '/hpc/group/vossenlab/kam264/Qmom2'
    if iii==3:
        root = '/hpc/group/vossenlab/kam264/Qmom3'   
    dataset = MyOwnDataset(
            root,
            transform=None, #T.Compose([T.ToUndirected(),T.KNNGraph(k=6)]),
            pre_transform=None,
            pre_filter=None
        ) 
    dataset
    model = GCN(dataset.num_node_features,64,2)
    fracs = [0.9, 0.08, 0.02] #percent of dataset used for training testing and validatoin 80%,10%,10% #NOTE: SHOULD CHECK np.sum(fracs) == 1 and len(fracs)==3
    fracs = [torch.sum(torch.tensor(fracs[:idx])) for idx in range(1,len(fracs)+1)] #get the indexes for training ... parts to use. 
    #print(fracs)
    split1, split2 = [int(len(dataset)*frac) for frac in fracs[:-1]] 
    train_dataset = dataset[:split1]
    val_dataset = dataset[split1:split2]
    test_dataset = dataset[split2:]   
     
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}') 

    from torch_geometric.loader import DataLoader 
    #from torch.utils.data import WeightedRandomSampler

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)#, drop_last=True)
    val_loader = DataLoader(val_dataset,  batch_size=batch_size,num_workers=8,  shuffle=True)
    model = GCN(dataset.num_node_features, dataset.num_classes, 1).to(device) #initiate the model, #2 is the number of outputs here is 2 as pion_z, proton_z 
    model = model.to(device)  
    del dataset
    del train_dataset
    del val_dataset
    del test_dataset 
    optimizer = torch.optim.Adam(model.parameters(), lr= LR)  
    
    losss = RMSELoss  
    nepochs = 800     
    
    for epoch in range(nepochs):  
  
        #print("BEFORE TRAIN()")                                                                                                                                                                                                                  
        train() 
        #print("BEFORE TEST(TRAIN_LOADER)")                                                                                                                                                                                                       
        #train_mse, train_rmse, train_mse_pi, train_rmse_pi, train_mse_p, train_rmse_p = test(train_loader)
        train_mse = test(train_loader)

        train_metrics['mse'].append(train_mse) 

        vall_mse =test(val_loader) 


        vall_metrics['mse'].append(vall_mse) 

        if epoch%9==0:
            print("Epoch ",epoch," Train mse: ",train_mse)
            print("Epoch ",epoch," Validation mse: ",vall_mse)
        if epoch==(nepochs-1):
            #PATH = '/work/clas12/users/mfmce/CLAS12_Lambda_resolution_REU_2023/model_best_auc.pt' 

            #a = print_out()
            #b=  print_outb() 
            print("Epoch ",epoch," Train mse: ",train_mse)
            print("Epoch ",epoch," Validation mse: ",vall_mse) 

root = '/hpc/group/vossenlab/kam264/Hmom4' 
#root = '/hpc/group/vossenlab/kam264/QuatorL_Last'
dataset = MyOwnDataset(
        root,
        transform=None, #T.Compose([T.ToUndirected(),T.KNNGraph(k=6)]),
        pre_transform=None,
        pre_filter=None
    )    

model.eval() #initailize the model                                                                                                                                                                                                        
#for i, data in tqdm(enumerate(train_loader)): #perhaps tqdm(enumerate(train_loader)), i is index, data jsut moves through all the dtaa in trainingg      

outt= []  
pi_mas = []
#for i,data in enumerate(dataset): 
for i,data in enumerate(dataset):
    data = data.to(device) #switch to GPU                                                                                                                                                                                                 
    #optimizer.zero_grad() #                                                                                                                                                                                                               
    out = model(data.x, data.edge_index, data.batch).to(device)  # Perform a single forward pass                                                                                                                                          
    out = out.cpu()
    #print(out) 
    #print(data.elc) 

   # neww = []
    #for j in range(len(data)):
        #neww.append(out[j].item() -data.elc[j][0])
     #   neww.append(out[j].item() -data.elc[j])
    #outt+=[[neww]]
    #outt+=[[out[0][0].item()-data.elc[0]]] 
    outt+=[[out[0][0].item()-data.elc[0]]]
    for j in range(0,int(len(data.imass))): 
        pi_mas.append(data.imass[j] ) 
#return outt  
#b= print_outbb() 
b=outt 