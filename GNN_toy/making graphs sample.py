import numpy as np  
import pandas as pd     
import uproot as ur
import awkward as ak 
import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader, random_split
import torch_geometric as tg

from torch_geometric.data import Data as dt

from torch_geometric.data import Dataset, download_url, InMemoryDataset

#These are the fucntions for calculting the commulants
def Qmoment(a, n):   
    return np.sum(np.exp(1j*n*a))  
#coorelation 2, 4 with no subevents
def corrilation_4(phi,pt,  n, momentum_cut = 0):  
    phi = phi[pt>momentum_cut]  
    M=len(phi) 
    if M<=3:
        #return np.nan 
        return 1234
    else: 
        Qn = Qmoment(phi, n) 
        Q2n = Qmoment(phi, 2*n)   
        demoninator = M*(M-1)*(M-2)*(M-3)
        first = np.abs(Qn)**4+np.abs(Q2n)**2-2*np.real(Q2n*np.conjugate(Qn)*np.conjugate(Qn))
        second = 2*(M-2)*np.abs(Qn)**2-M*(M-3)
        return (first-2*second)/demoninator  
def corrilation_2(phi, pt, n, momentum_cut = 0):   
    phi = phi[pt>momentum_cut]
    M= len(phi)  
    if M<=1:
        #return np.nan
        return  1234
    else:
        return (np.abs(Qmoment(phi, n))**2-M)/( (M-1)*M ) 
        
# for saving the new graph datasets
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

#reads in the root file
file = ur.open(r"1merged_2pt.root") # shouldnt have to chnge this if on the campus cluster, but its just the path 


# List all keys (e.g., trees, histograms)    
#print(file.keys())    
  
# Access a TTree    
startt = 0 
endd = 1000
tree = file["jet_tree;1"]   #I have no clue why its jet_tree;1;1
#org is event, particle number
vtrackphi  = tree['vtrackphi'].array(entry_start = startt, entry_stop = endd)  #phi positiion of each particle
weights = tree['weight'].array(entry_start = startt, entry_stop = endd) #weight of each event
ptt = tree['vtrackpt'].array(entry_start = startt, entry_stop = endd) #momentum of each partilce
#vtrackpt1  = tree['vtrackpt'].array(entry_start = startt, entry_stop = endd)
rapitity = tree['vtracketa'].array(entry_start = startt, entry_stop = endd) #rapidity of each particle
#Changes to lsit of arrays instead of list of lists
phii = [np.array(x) for x in vtrackphi]
eta = [np.array(x) for x in rapitity]
pt = [np.array(x) for x in ptt]
#normalize weights, not strictly needed, but helps with numerical error
weights = weights/np.sum(weights)

#Calculates hte correlations

#calcualtes the 2, 4 particle correlations for each event
cor2 = []
cor4 = []
for i in range(0, len(phii)):
    cor2.append(corrilation_2(phii[i], pt[i], 2))
    cor4.append(corrilation_4(phii[i], pt[i], 2)) 
# Function that turns it into graphs
def build_graphs(phi, pt, eta, weight, true_cor2, true_cor4):
    #more complicated input, each vertex is phi, pt, eta
    #x = torch.tensor(list(zip(phi, pt, eta)), dtype=torch.float)
    #input is jsut list of phi. so each vertex is jsut phi
    x = torch.tensor(phi, dtype=torch.float)
    #print(x)
    
    N = x.size(0)
    
    #making the edges, here it is jsut 1 to 2, 2 to3 , ..., then later it is made to go the opposite way, very simple method
    # Edge list: i -> i+1
    edge_index = torch.tensor(
        [
            list(range(N - 1)),      # source nodes
            list(range(1, N))        # target nodes
        ],
        dtype=torch.long
    )
    #for fully conected graphs
    '''edge_index = torch.combinations(torch.arange(N), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)'''

    
    
# Optional: make edges bidirectional (usually better for GNNs)
    edge_index = torch.cat(
        [edge_index, edge_index.flip(0)],
        dim=1
    )
    #print(edge_index)
    
    #the 'True' values
    y = [ true_cor4]
    #combinning them all into a data type
    data = dt(x=x, edge_index=edge_index, y = torch.tensor(y))
    #this is adding extra saving on the end, so that we save important stuff like the event weight and can access it later
    data.weight = torch.tensor([weight], dtype = torch.float)
    data.multiplicity = torch.tensor([len(phi)], dtype = torch.float)
    return data

graph = []

#apply to every event
for i in range(0, len(phii)):
    if (cor4[i]!= 1234): #if cor4 is not defined skip the event
    #phi, eta, pt = event["phi"], event["eta"], event["pt"]
        graph.append(build_graphs(phii[i], pt[i], eta[i],weights[i],  cor2[i], cor4[i]))
graph
#save, to new directory, under name first, everytime you make a new graph change the name of first, IT WILL NOT OVERWRITE IT
root  = '~/GNN_graphs/first'
MyDataset(root, graph)