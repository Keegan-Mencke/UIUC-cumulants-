import numpy as np  
import pandas as pd     
import uproot as ur      

#def Qn
#paper 1 https://arxiv.org/pdf/1010.0233 
#paper 2 https://arxiv.org/pdf/1701.03830
def Qmoment(a, n):   
    return np.sum(np.exp(1j*n*a))  
#coorelation 2, 4 with no subevents
def corrilation_4(phi,pt,  n, momentum_cut = 0):  
    phi = phi[pt>momentum_cut]  
    M=len(phi) 
    if M<=3:
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
#cor 4 for 4 subevents, note that cor 2 does not exist. this function also has the weights and #2 subevents cor2 and cor 4 
def sub_cors(phi, rapity, weight, pt,  n, momentum_cut = 0):
    #a, b, c, d based on rapity ranges
    phi = phi[pt>momentum_cut]
    rapity = rapity[pt>momentum_cut]
    maska = (rapity >= -2.4) & (rapity < -1.2)
    maskb = (rapity >= -1.4 )& (rapity<0)
    maskc = (rapity >= 0 )& (rapity<1.2 ) 
    maskd = (rapity>=1.2) & (rapity <=2.4 ) 
    phi_a = phi[maska] 
    phi_b = phi[maskb]
    phi_c = phi[maskc]
    phi_d = phi[maskd]
    M_a = len(phi_a)
    M_b = len(phi_b)
    M_c = len(phi_c)
    M_d = len(phi_d)
    if M_a+M_b<=1 or M_c+M_d<=1:
        return 1234, 1234, 1234, 1234, 1234, 1234, 1234, weight
    elif M_a==0 or M_b==0 or M_c==0 or M_d==0:
        #defining stuff for 2 subevents
        phi_aa = np.concatenate((phi_a, phi_b)) #subevent a
        phi_bb = np.concatenate((phi_c, phi_d)) #subevent b
        cor2_2e = Qmoment(phi_aa, n) *np.conjugate(Qmoment(phi_bb, n))/len(phi_aa)/len( phi_bb) #eq 19 paper 2
        cor4_2e = ((Qmoment(phi_aa, n)**2- Qmoment(phi_aa, 2*n))*np.conjugate(Qmoment(phi_bb, n)**2-Qmoment(phi_bb, 2*n)) #eq 20 paper 2
                       /(len( phi_aa)*(len(phi_aa)-1)*len( phi_bb)*(len( phi_bb)-1)) )
        return cor4_2e, cor2_2e, 1234, 1234, 1234, 1234, 1234, weight
    else:
        Qna = Qmoment(phi_a, n)
        Qnb = Qmoment(phi_b, n)
        Qncc = np.conjugate(Qmoment(phi_c, n) )
        Qndc = np.conjugate(Qmoment(phi_d, n) )
        #<4>, euqation 44a in paper 2
        cor4_4e = (Qna* Qnb *Qncc*Qndc 
                /M_a/M_b/M_c/M_d)
        #these are need to define Cn{4} in second line eq 44 in paper 2. these take the place of <2>
        #note their is not <2> here
        cor_ac = Qna*Qncc/M_a/M_c
        cor_bd = Qnb*Qndc/M_b/M_d
        cor_ad = Qna*Qndc/M_a/M_d
        cor_bc = Qnb*Qncc/M_b/M_c
    
        #defining stuff for 2 subevents
        phi_aa = np.concatenate((phi_a, phi_b)) #subevent a
        phi_bb = np.concatenate((phi_c, phi_d)) #subevent b
        cor2_2e = Qmoment(phi_aa, n) *np.conjugate(Qmoment(phi_bb, n))/len(phi_aa)/len( phi_bb) #eq 19 paper 2
        cor4_2e = ((Qmoment(phi_aa, n)**2- Qmoment(phi_aa, 2*n))*np.conjugate(Qmoment(phi_bb, n)**2-Qmoment(phi_bb, 2*n)) #eq 20 paper 2
                       /(len( phi_aa)*(len(phi_aa)-1)*len( phi_bb)*(len( phi_bb)-1)) )
        return cor4_2e, cor2_2e, cor4_4e, cor_ac, cor_bd, cor_ad, cor_bc, weight 

from collections import defaultdict    

def apply_and_bin_vectorized(arr_list, bin_size, n, momentum_cut): 
    # Convert to object array (fast iteration, no Python list overhead) 
    arrs = np.array(arr_list, dtype=object)

    # Vectorized extraction of lengths
   # lengths = np.vectorize(len)(arrs[0])        # e.g. array([12, 7, 33, ...])
    lengths = np.array([len(a[0]) for a in arr_list] )
    # Compute the bin index for each length
    # 5→1, 6→1, ..., 9→1
    # 10→2, 11→2, ...
    #bins = ((lengths - 5) // 5)
    bins = ((lengths - bin_size) // bin_size))
    # Storage: dict {bin_start_length: [results]} 
    #all the stuff needed for 0 subevents 
    corrilation4e0 = defaultdict(list) 
    corrilation2e0 = defaultdict(list) 
    
    #all stuff for 2 subevents
    corrilation4e2 = defaultdict(list) 
    corrilation2e2 = defaultdict(list) 
    
    #all stuff needed for 4 subevents
    corrilation4e4 = defaultdict(list) 
    corr_ac = defaultdict(list) 
    corr_bd = defaultdict(list)
    corr_bc = defaultdict(list)
    corr_ad = defaultdict(list)
    
    weightss = defaultdict(list)
 
    for arr, b in zip(arrs, bins): 
        bin_start = bin_size + bin_size * b     # convert bin index → actual group range start
        #no subevents 
        corrilation4e0[bin_start].append(corrilation_4(arr[0], arr[3],  n, momentum_cut = momentum_cut)) 
        corrilation2e0[bin_start].append(corrilation_2(arr[0], arr[3], n, momentum_cut =momentum_cut)) 

        #4 subevents and weights, new function, all in one, and 2 subevents
        cor4e2_shit, cor2e2_shit, cor4e4_shit, cor_ac_shit, cor_bd_shit, cor_ad_shit, cor_bc_shit, weight_shit = sub_cors(
            arr[0],arr[1], arr[2], arr[3], n, momentum_cut =momentum_cut)
        #2 subevents
        corrilation4e2[bin_start].append(cor4e2_shit)
        corrilation2e2[bin_start].append(cor2e2_shit)

        #4 subevents
        corrilation4e4[bin_start].append(cor4e4_shit )  
        corr_ac[bin_start].append(cor_ac_shit)
        corr_bd[bin_start].append(cor_bd_shit)
        corr_ad[bin_start].append(cor_ad_shit)
        corr_bc[bin_start].append(cor_bc_shit)
        weightss[bin_start].append(weight_shit )  

    return corrilation4e0, corrilation2e0, corrilation4e2, corrilation2e2, corrilation4e4,  corr_ac, corr_bd, corr_ad, corr_bc,  weightss, lengths    

file = ur.open(r"1merged_2pt.root") #  size, min bias, new dataset 


# List all keys (e.g., trees, histograms)    
#print(file.keys())    
  
# Access a TTree    
startt = 0 
endd = -1
tree = file["jet_tree;1"]   #I have no clue why its jet_tree;1;1
vtrackphi  = tree['vtrackphi'].array(entry_start = startt, entry_stop = endd)  # Replace with actual tree name
weights = tree['weight'].array(entry_start = startt, entry_stop = endd)
ptt = tree['vtrackpt'].array(entry_start = startt, entry_stop = endd)
rapitity = tree['vtracketa'].array(entry_start = startt, entry_stop = endd)
phii = [np.array(x) for x in vtrackphi]
eta = [np.array(x) for x in rapitity]
pt = [np.array(x) for x in ptt]
weights = weights/np.sum(weights)
testing12 = [[x,y, z, w] for x,y, z, w in zip(phii,eta , weights, pt)]  

for momentum in np.array([0.0, 1.0, 5.0]): 
    out_corrilation4e0, out_corrilation2e0, out_corrilation4e2, out_corrilation2e2, out_corrilation4e4, out_cor_ac,  out_cor_bd, out_cor_ad, out_cor_bc, weightt, multiplicity= apply_and_bin_vectorized(testing12 , 10, 2, momentum)   
  
    colscor4e0 = {}
    colscor2e0 = {}
    colscor4e2 = {}
    colscor2e2 = {}
    colscor4e4 = {}
    cols_cnac = {}
    cols_cnbd = {}
    cols_cnbc = {}
    cols_cnad = {} 
    cols_wweight = {} 
    
    for i in range(50, 250, 10):
        title = str(i)
    
        cor4e0 = np.array(out_corrilation4e0[i])
        cor2e0 = np.array(out_corrilation2e0[i])
        cor4e2 = np.array(out_corrilation4e2[i]).real
        cor2e2 = np.array(out_corrilation2e2[i]).real
        cor4e4 = np.array(out_corrilation4e4[i]).real
       
    
        cnac_idk = np.array(out_cor_ac[i]).real
        cnbd_idk = np.array(out_cor_bd[i]).real
        cnbc_idk = np.array(out_cor_bc[i]).real
        cnad_idk = np.array(out_cor_ad[i]).real
        wweight  = np.array(weightt[i])
    
        colscor4e0[title] = pd.Series(cor4e0)
        colscor2e0[title] = pd.Series(cor2e0)
        colscor4e2[title] = pd.Series(cor4e2)
        colscor2e2[title] = pd.Series(cor2e2)
        colscor4e4[title] = pd.Series(cor4e4)
    
        cols_cnac[title] = pd.Series(cnac_idk)
        cols_cnbd[title] = pd.Series(cnbd_idk)
        cols_cnbc[title] = pd.Series(cnbc_idk)
        cols_cnad[title] = pd.Series(cnad_idk)
        cols_wweight[title] = pd.Series(wweight)
    
    # -------------------------
    # WRITE EACH VARIABLE TO ITS OWN csv 
    # ------------------------- 
    
    pd.DataFrame(colscor4e0).fillna(0).to_csv("ptcor4e0_"+str(momentum)+".csv", index=False) 
    pd.DataFrame(colscor2e0).fillna(0).to_csv("ptcor2e0_"+str(momentum)+".csv", index=False)
    pd.DataFrame(colscor4e2).fillna(0).to_csv("ptcor4e2_"+str(momentum)+".csv", index=False)
    pd.DataFrame(colscor2e2).fillna(0).to_csv("ptcor2e2_"+str(momentum)+".csv", index=False)
    pd.DataFrame(colscor4e4).fillna(0).to_csv("ptcor4e4_"+str(momentum)+".csv", index=False)
    
    pd.DataFrame(cols_cnac).fillna(0).to_csv("ptcnac_idk"+str(momentum)+".csv", index=False)
    pd.DataFrame(cols_cnbd).fillna(0).to_csv("ptcnbd_idk"+str(momentum)+".csv", index=False)
    pd.DataFrame(cols_cnbc).fillna(0).to_csv("ptcnbc_idk"+str(momentum)+".csv", index=False)
    pd.DataFrame(cols_cnad).fillna(0).to_csv("ptcnad_idk"+str(momentum)+".csv", index=False)
    pd.DataFrame(cols_wweight).fillna(0).to_csv("ptwweight"+str(momentum)+".csv", index=False )  

for momentum in np.array([0.0, 1.0, 5.0]):    
    #for j in range(2, 4):  
    for j in range(2, 4):  
    
        file = ur.open(str(j)+"merged_2pt.root") #  size, min bias, new datasetfile = ur.open(r"1merged_2pt.root")
        #file = ur.open(r"1merged_2pt.root")
         
        # Access a TTree     
        startt = 0 
        endd = -1 
        tree = file["jet_tree;1"]   #I have no clue why its jet_tree;1;1
        vtrackphi  = tree['vtrackphi'].array(entry_start = startt, entry_stop = endd)  # Replace with actual tree name
        weights = tree['weight'].array(entry_start = startt, entry_stop = endd)
        ptt = tree['vtrackpt'].array(entry_start = startt, entry_stop = endd)
        #vtrackpt1  = tree['vtrackpt'].array(entry_start = startt, entry_stop = endd)
        rapitity = tree['vtracketa'].array(entry_start = startt, entry_stop = endd)
        phii = [np.array(x) for x in vtrackphi]
        eta = [np.array(x) for x in rapitity]
        pt = [np.array(x) for x in ptt]
        weights = weights/np.sum(weights) 
        testing12 = [[x,y, z, w] for x,y, z, w in zip(phii,eta , weights, pt)] 
        out_corrilation4e0, out_corrilation2e0, out_corrilation4e2, out_corrilation2e2, out_corrilation4e4, out_cor_ac,  out_cor_bd, out_cor_ad, out_cor_bc, weightt, multiplicity= apply_and_bin_vectorized(testing12 , 10, 2, momentum)
        
        import numpy as np 
        import pandas as pd
        base_length = len(out_corrilation4e0[10])
        colscor4e0 = {}
        colscor2e0 = {}
        colscor4e2 = {}
        colscor2e2 = {}
        colscor4e4 = {}
        cols_cnac = {}
        cols_cnbd = {}
        cols_cnbc = {}
        cols_cnad = {}
        cols_wweight = {}
        def update_corr_csv(
            csv_file,
            out_corr,
            cols_dict,
            base_length, 
            col_range=range(50, 250, 10)
        ):
            # Read existing CSV
            old = pd.read_csv(csv_file)
            old_np = old.to_numpy().T
        
            # Update each column
            for i in col_range:
        
                title = str(i)
        
                # Trim trailing zeros from old column
                #old_trimmed = np.trim_zeros(old_np[i // 10 - 1], trim='b')
                old_trimmed = np.trim_zeros(old_np[i // 10 - 5], trim='b')
        
                # New correlation data
                new_vals = np.array(out_corr[i]).real
                N = base_length-len(new_vals)
                if i==10 and N>0:
                    new_vals = new_vals.resize(new_vals.size + N, refcheck=False)
        
                # Append
                combined = np.append(new_vals, old_trimmed)
        
                cols_dict[title] = pd.Series(combined)
        
            # Save back with padding filled as 0
            pd.DataFrame(cols_dict).fillna(0).to_csv(csv_file, index=False) 
        update_corr_csv("ptcor4e0_"+str(momentum)+".csv", out_corrilation4e0, colscor4e0, base_length)
        update_corr_csv("ptcor2e0_"+str(momentum)+".csv", out_corrilation2e0, colscor2e0, base_length)
        update_corr_csv("ptcor4e2_"+str(momentum)+".csv", out_corrilation4e2, colscor4e2, base_length)
        update_corr_csv("ptcor2e2_"+str(momentum)+".csv", out_corrilation2e2, colscor2e2, base_length)
        update_corr_csv("ptcor4e4_"+str(momentum)+".csv", out_corrilation4e4, colscor4e4, base_length)
        
        update_corr_csv("ptcnac_idk"+str(momentum)+".csv", out_cor_ac, cols_cnac, base_length)
        update_corr_csv("ptcnbd_idk"+str(momentum)+".csv", out_cor_bd, cols_cnbd, base_length) 
        update_corr_csv("ptcnbc_idk"+str(momentum)+".csv", out_cor_bc, cols_cnbc, base_length)
        update_corr_csv("ptcnad_idk"+str(momentum)+".csv", out_cor_ad, cols_cnad, base_length)
        update_corr_csv("ptwweight"+str(momentum)+".csv", weightt, cols_wweight, base_length)  
        #j=1
        print('done'+str(j))  