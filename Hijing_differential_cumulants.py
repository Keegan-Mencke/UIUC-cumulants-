import numpy as np   
import pandas as pd      
import uproot as ur  

file = ur.open(r"data/1p_pbHijing_1M.root") #, not events in mult range 60-120 is about 640000
#file = ur.open(r"data/pbpb_mb_500k.root") 
#file = ur.open(r"data/p_pb_jet.root") 
#file = ur.open(r"data/2merged_2pt.root")  

# List all keys (e.g., trees, histograms)    
print(file.keys())    
  
# Access a TTree     
startt = 0 
endd = 100000
tree = file["tree;1"]   #I have no clue why its jet_tree;1;1
phi  = tree['phi'].array(entry_start = startt, entry_stop = endd)  # Replace with actual tree name
pt = tree['pt'].array(entry_start = startt, entry_stop = endd)
eta = tree['eta'].array(entry_start = startt, entry_stop = endd)
weights = tree['weight'].array(entry_start = startt, entry_stop = endd) 

# tree = file["jet_tree;1"]   #I have no clue why its jet_tree;1;1
# phi  = tree['vtrackphi'].array(entry_start = startt, entry_stop = endd)  # Replace with actual tree name
# weights = tree['weight'].array(entry_start = startt, entry_stop = endd)
# pt = tree['vtrackpt'].array(entry_start = startt, entry_stop = endd)


#eta = tree['eta'].array(entry_start = startt, entry_stop = endd)
phi = [np.array(x) for x in phi]
eta = [np.array(x) for x in eta]
pt = [np.array(x) for x in pt]
weights = weights/np.sum(weights)
testing12 = [[x, z, w, y] for x, z, w, y in zip(phi, weights, pt, eta)]
del pt
del file
del eta
del phi

import numpy as np  
import pandas as pd     
import uproot as ur      

#def Qn
#paper 1 https://arxiv.org/pdf/1010.0233 
#paper 2 https://arxiv.org/pdf/1701.03830
def Qmoment(a, n):   
    return np.sum(np.exp(1j*n*a)).item() 
    
# these functions now return the correlation and then the weight which is basically number of combinations
#coorelation 2, 4 with no subevents    
def corrilation_4(phi,  n, momentum_cut = 0):  #eq. 18, weights by eq. 10
    M=len(phi) 
    if M<=3:
        #return np.nan 
        return -1234, -1234
    else: 
        Qn = Qmoment(phi, n) 
        Q2n = Qmoment(phi, 2*n)   
        demoninator = M*(M-1)*(M-2)*(M-3)
        first = np.abs(Qn)**4+np.abs(Q2n)**2-2*np.real(Q2n*np.conjugate(Qn)*np.conjugate(Qn))
        second = 2*(M-2)*np.abs(Qn)**2-M*(M-3)
        return (first-2*second)/demoninator, demoninator  
def corrilation_2(phi, n, momentum_cut = 0):   #eq. 16, weights by eq. 9
    M= len(phi)  
    if M<=1:
        #return np.nan
        return  -1234, -1234
    else:
        return (np.abs(Qmoment(phi, n))**2-M)/( (M-1)*M ), (M-1)*M

def dcor_4(phi, pt, n, POI_start=1 , POI_end = 2): #equ 32, with M_q=q_n =0, weights by eq. 25
    phi = np.array(phi); pt = np.array(pt)
    mask = (pt>=POI_start) & (pt<= POI_end )
    POI = phi[mask]
    Ref = phi[pt<POI_start]
    mp = len(POI)
    M=len(Ref) 
    if (M<=2) or  (mp==0):
        #return np.nan 
        return -1234, -1234
    Qnc = np.conjugate(Qmoment(Ref, n) ) #conjugate of Qn is what is needed the most
    Qn = Qmoment(Ref, n)  #conjugate of Qn is what is needed the most
    Q2n = Qmoment(Ref, 2*n)
    pn = Qmoment(POI, n) 
    return np.real( (pn*Qn*Qnc*Qnc-pn*Qn*Q2n-2*M*pn*Qnc+2*pn*Qnc) / (mp*M*(M-1)*(M-2))), mp*M*(M-1)*(M-2) 
def dcor_2(phi, pt, n, POI_start=1 , POI_end = 2): #equ 28, with M_q=q_n =0, weights by eq. 24
    phi = np.array(phi); pt = np.array(pt)
    mask = (pt>=POI_start) & (pt<=POI_end )
    POI = phi[mask]
    Ref = phi[pt<POI_start]
    mp = len(POI)
    M=len(Ref) 
    if (M==0) or (mp==0):
        #return np.nan 
        return -1234, -1234
    return np.real(np.conjugate(Qmoment(Ref, n))*Qmoment(POI, n)/mp/M), mp*M
    
#this was fornudging v2 to teh v2 we want to see if cor4 is negative then, was from ai, but if it works it works
def compute_v2_truth(phi, psi2=0.0):
    """Compute v2 = <cos(2(phi-Psi2))>."""
    phi = np.asarray(phi)
    return np.mean(np.cos(2 * (phi - psi2)))
def nudge_v2(phi, target_v2=0.5, psi2=0.0,
             tol=1e-12, max_iter=50):
    """
    Adjust phi by the smallest L2 change needed to reach a target v2.
    Parameters
    ----------
    phi : array_like
        Particle azimuths.
    target_v2 : float
        Desired v2.
    psi2 : float
        Event-plane angle.
    """
    phi = np.asarray(phi, dtype=float).copy()
    M = len(phi)
    if M == 0:
        return phi
    for _ in range(max_iter):
        current_v2 = compute_v2_truth(phi, psi2)
        dv = target_v2 - current_v2
        if abs(dv) < tol:
            break
        # dv2/dphi_i
        g = -(2.0 / M) * np.sin(2 * (phi - psi2))
        g2 = np.dot(g, g)
        if g2 < 1e-20:
            raise RuntimeError("Gradient vanished.")
        # Minimum-norm correction
        dphi = (dv / g2) * g
        phi += dphi
        # keep angles in [0,2π)
        phi %= 2*np.pi
    return phi
    
def cors(phi, weight, pt,  n, POI_start=1 , POI_end = 2, momentum_cut = 0, nudge = False):
    M = len(phi)
    if (nudge ==True):
        phi = nudge_v2(phi)
    phi = phi[pt>momentum_cut]
    
    cor2, cor2w = corrilation_2(phi, n)
    cor4, cor4w = corrilation_4(phi, n)
    dcor2, dcor2w = dcor_2(phi, pt, n, POI_start=1,  POI_end = 2)
    dcor4, dcor4w = dcor_4(phi, pt, n, POI_start=1,  POI_end = 2)
    d =  [dcor4, dcor4w, dcor2 , dcor2w ]
    c= [cor4, cor4w, cor2, cor2w]
    return d, c# dcor4, dcor2, cor4, cor2
    
'''def REF_POI_togehter_0sub(phi, weight, pt,  n, POI_cut=1, momentum_cut = 0, nudge = False):
    M = len(phi)
    if (nudge ==True):
        phi = nudge_v2(phi)
    phi = phi[pt>momentum_cut]
    #def dcor_4(phi, pt, n, POI_cut): #equ 32, with M_q=q_n =0, weights by eq. 25
    phi = np.array(phi); pt = np.array(pt)
    mask = (pt>=POI_cut) & (pt<=POI_cut+1 )
    POI = phi[mask]
    Ref = phi[pt<POI_cut]
    mp = len(POI)
    M=len(Ref) 
        if (M<=2) or  (mp==0):
            #return np.nan 
            return -1234, -1234
        Qnc = np.conjugate(Qmoment(Ref, n) ) #conjugate of Qn is what is needed the most
        Qn = Qmoment(Ref, n)  #conjugate of Qn is what is needed the most
        Q2n = Qmoment(Ref, 2*n)
        pn = Qmoment(POI, n) 
        return np.real( (pn*Qn*Qnc*Qnc-pn*Qn*Q2n-2*M*pn*Qnc+2*pn*Qnc) / (mp*M*(M-1)*(M-2))), mp*M*(M-1)*(M-2) 
    def dcor_2(phi, pt, n, POI_cut): #equ 28, with M_q=q_n =0, weights by eq. 24
        
        if (M==0) or (mp==0):
            #return np.nan 
            return -1234, -1234
        return np.real(np.conjugate(Qmoment(Ref, n))*Qmoment(POI, n)/mp/M), mp*M

    dcor2, dcor2w = dcor_2(phi, pt, n, POI_cut)
    dcor4, dcor4w = dcor_4(phi, pt, n, POI_cut)
    d =  [dcor4, dcor4w, dcor2 , dcor2w ]
    c= [cor4, cor4w, cor2, cor2w]
    return d, c# dcor4, dcor2, cor4, cor2'''

def sub2_diff_cors(phi, weight, pt, rapity, n,  POI_start=1 , POI_end = 2):
    #the correlators are given by eq 19 and 20 in this paper https://arxiv.org/pdf/1701.03830. cummulants are found in the cms paper
    maska = (rapity >= -2.4) & (rapity < 0)
    maskb = (rapity >= 0 )& (rapity<=2.4)
    phi_a = phi[maska] 
    phi_b = phi[maskb]

    pt_a = pt[maska] 
    pt_b = pt[maskb]

    POI_a = phi_a[(pt_a>=POI_start) & (pt_a<=POI_end)]
    POI_b = phi_b[(pt_b>=POI_start) & (pt_b<=POI_end)] 

    M_a = len(phi_a); M_b = len(phi_b)
    m_a = len(POI_a); m_b = len(POI_b)

    Qa = Qmoment(phi_a, n); Qb = Qmoment(phi_b, n); Q2a = Qmoment(phi_a, 2*n); Q2b = Qmoment(phi_b, 2*n)
    pa = Qmoment(POI_a, n); pb = Qmoment(POI_b, n); p2a = Qmoment(POI_a, 2*n); p2b = Qmoment(POI_b, 2*n)
    if (m_a<=1 or M_b<=1):
        dcor2a= -1234 *np.ones(6)
    else:
        dcor2a =  [(pa**2-p2a)*np.conjugate(Qb**2-Q2b)/(m_a*(m_a-1)*M_b*(M_b-1)), m_a*(m_a-1)*M_b*(M_b-1), pa*np.conjugate(Qb)/m_a/M_b, m_a*M_b,
                Qa*np.conjugate(Qb)/M_a/M_b, M_a*M_b]
    if (M_a<=1 or m_b<=1):
        dcor2b= -1234 *np.ones(6)
    else:
        dcor2b =  [(Qa**2-Q2a)*np.conjugate(pb**2-p2b)/(M_a*(M_a-1)*m_b*(m_b-1)), M_a*(M_a-1)*m_b*(m_b-1), Qa*np.conjugate(pb)/M_a/m_b, M_a*m_b,
                Qa*np.conjugate(Qb)/M_a/M_b, M_a*M_b]
    return dcor2a, dcor2b
        
    
    
def sub4_diff_cors(phi, weight, pt, rapity, n,  POI_start=1 , POI_end = 2):  
    #rapity = np.array(rapity)
    maska = (rapity >= -2.4) & (rapity < -1.2)
    maskb = (rapity >= -1.2 )& (rapity<0)
    maskc = (rapity >= 0 )& (rapity<1.2 ) 
    maskd = (rapity>=1.2) & (rapity <=2.4 ) 
    phi_a = phi[maska] 
    phi_b = phi[maskb]
    phi_c = phi[maskc]
    phi_d = phi[maskd]
    
    pt_a = pt[maska] 
    pt_b = pt[maskb]
    pt_c = pt[maskc]
    pt_d = pt[maskd]

    POI_a = phi_a[(pt_a>=POI_start) & (pt_a<=POI_end)]
    POI_b = phi_b[(pt_b>=POI_start) & (pt_b<=POI_end)] 
    POI_c = phi_c[(pt_c>=POI_start) & (pt_c<=POI_end)]
    POI_d = phi_d[(pt_d>=POI_start) & (pt_d<=POI_end)]

    M_a = len(phi_a); M_b = len(phi_b); M_c = len(phi_c); M_d = len(phi_d);
    m_a = len(POI_a); m_b = len(POI_b); m_c = len(POI_c); m_d = len(POI_d);

    Qa = Qmoment(phi_a, n); Qb = Qmoment(phi_b, n); Qc = Qmoment(phi_c, n); Qd = Qmoment(phi_d, n)
    pa = Qmoment(POI_a, n); pb = Qmoment(POI_b, n); pc = Qmoment(POI_c, n); pd = Qmoment(POI_d, n)
    if ((m_a==0)|(M_b==0)|(M_c==0)|(M_d==0)):
        dcor4a=-1234 *np.ones(10)
    else:
        dcor4a = [pa*Qb *np.conjugate(Qc)*np.conjugate( Qd)/(m_a*M_b*M_c*M_d), m_a*M_b*M_c*M_d, pa*np.conjugate(Qc)/(m_a*M_c), m_a*M_c, Qb*np.conjugate( Qd)/(M_b*M_d), M_b*M_d, 
                 pa*np.conjugate(Qd)/(m_a*M_d), m_a*M_d, Qb*np.conjugate( Qc)/(M_b*M_c), M_b*M_c]
        
    if ((M_a==0)|(m_b==0)|(M_c==0)|(M_d==0)):
        dcor4b=-1234*np.ones(10)
    else:
        dcor4b = [Qa*pb *np.conjugate(Qc)*np.conjugate( Qd)/(M_a*m_b*M_c*M_d), M_a*m_b*M_c*M_d, Qa*np.conjugate(Qc)/( M_a*M_c), M_a*M_c, pb*np.conjugate( Qd)/(m_b*M_d), m_b*M_d, 
                 Qa*np.conjugate(Qd)/(M_a*M_d), M_a*M_d, pb*np.conjugate( Qc)/(m_b*M_c), m_b*M_c]
        
    if ((M_a==0)|(M_b==0)|(m_c==0)|(M_d==0)):
        dcor4c=-1234*np.ones(10)
    else:
        dcor4c = [Qa*Qb *np.conjugate(pc)*np.conjugate( Qd)/(M_a*M_b*m_c*M_d), M_a*M_b*m_c*M_d,  Qa*np.conjugate(pc)/(M_a*m_c), M_a*m_c, Qb*np.conjugate( Qd)/(M_b*M_d), M_b*M_d, 
                 Qa*np.conjugate(Qd)/(M_a*M_d), M_a*M_d, Qb*np.conjugate( pc)/(M_b*m_c), M_b*m_c]
        
    if ((M_a==0)|(M_b==0)|(M_c==0)|(m_d==0)):
        dcor4d=-1234*np.ones(10)
    else:
        dcor4d = [Qa*Qb *np.conjugate(Qc)*np.conjugate( pd)/( M_a*M_b*M_c*m_d), M_a*M_b*M_c*m_d, Qa*np.conjugate(Qc)/(M_a*M_c), M_a*M_c, Qb*np.conjugate( pd)/(M_b*m_d), M_b*m_d, 
                 Qa*np.conjugate(pd)/(M_a*m_d), M_a*m_d, Qb*np.conjugate( Qc)/( M_b*M_c), M_b*M_c]
    return dcor4a, dcor4b, dcor4c, dcor4d




from collections import defaultdict    

def mult_binning(arr_list, bin_size, n, momentum_cut = 0, nudge = False ): 
    arrs = np.array(arr_list, dtype=object)

    # Vectorized extraction of lengths
    lengths = np.array([len(a[0]) for a in arr_list] )

    # Compute the bin index for each length
    # 5→1, 6→1, ..., 9→1
    # 10→2, 11→2, ...
    bins = ((lengths - bin_size) // bin_size) 


    # Storage: dict {bin_start_length: [results]} 
    #all the stuff needed for 0 subevents 
    corrilation4e0 = defaultdict(list) 
    corrilation2e0 = defaultdict(list)
    dcorrilation4e0 = defaultdict(list) 
    dcorrilation2e0 = defaultdict(list) 
    
    weightss = defaultdict(list)
 
    for arr, b in zip(arrs, bins): 
        #if (b==185):
            

        bin_start = bin_size + bin_size * b     # convert bin index → actual group range start

        #no subevents 
        cor2, cor4, dcor2, dcor4, weight_shit, M = cors(arr[0], arr[1], arr[2], n, POI_cut=1, momentum_cut =  momentum_cut, nudge = nudge)
        corrilation4e0[bin_start].append(cor4) 
        corrilation2e0[bin_start].append(cor2) 
        dcorrilation4e0[bin_start].append(dcor4) 
        dcorrilation2e0[bin_start].append(dcor2) 

        #4 subevents and weights, new function, all in one, and 2 subevents
        '''cor4e2_shit, cor2e2_shit, cor4e4_shit, cor_ac_shit, cor_bd_shit, cor_ad_shit, cor_bc_shit, weight_shit = sub_cors(
            arr[0],arr[1], arr[2], arr[3], n, momentum_cut =momentum_cut)'''
        #2 subevents
        
        weightss[bin_start].append(weight_shit )  

    return corrilation4e0, corrilation2e0, dcorrilation4e0, dcorrilation2e0, weightss
#for older pt function with csv, see HIJing test
    
def pt_binning(arr_list, mult_range, n, POI_start=1 , POI_end = 2, momentum_cut=0, nudge=False):
    arrs = np.array(arr_list, dtype=object)

    # Vectorized extraction of lengths
    lengths = np.array([len(a[0]) for a in arr_list]) #lengths is mult
    

    mask = (lengths >= mult_range[0]) & (lengths <= mult_range[1])
    #print(mask)
    arrs = arrs[mask]
    print(len(arrs))
    

    # 10 lists for each subevent
    #these are the differential cumulaants (corelation 4) components and weights for 4 subevents. a is POI in subevent a, ....
    dcor4ae4 = [[] for _ in range(10)] 
    dcor4be4 = [[] for _ in range(10)]
    dcor4ce4 = [[] for _ in range(10)]
    dcor4de4 = [[] for _ in range(10)]
    #these are the differential cumulaants (corelation 4) components and weights for 2 subevents. a is POI in subevent a, ....
    dcor4ae2 = [[] for _ in range(6)]
    dcor4be2 = [[] for _ in range(6)]
    #these are components for differerential and standard cumulants for both 2, 4 correlations, no subevents
    dcore0 = [[] for _ in range(4)]
    core0 = [[] for _ in range(4)]

    

    #this for looops calculates all the needed correlators and weights. It then populates the storage functions
    for phi, weight, pt, eta in arrs:
   

        a, b, c, d = sub4_diff_cors(
            phi,
            weight,
            pt,
            eta,
            n,
            POI_start=POI_start, POI_end = POI_end
        )
        e, f = cors(phi, weight, pt, n,POI_start=POI_start, POI_end = POI_end) #returns  d =  [dcor4, dcor4w, dcor2 , dcor2w ], then d, c
        h, g = sub2_diff_cors(phi, weight,pt,eta,n,POI_start=POI_start, POI_end = POI_end)

        for i in range(10):
            dcor4ae4[i].append(np.real(a[i]))
            dcor4be4[i].append(np.real(b[i]))
            dcor4ce4[i].append(np.real(c[i]))
            dcor4de4[i].append(np.real(d[i]))
        for i in range(4):
            dcore0[i].append(np.real(e[i]))
            core0[i].append(np.real(f[i]))
        for i in range(6):
            dcor4ae2[i].append(np.real(h[i]))
            dcor4be2[i].append(np.real(g[i]))
    del arrs
    #following two functions calculate the dcummulants given the correlators. we then need to break out corelators blocks into 20 parts to feed to this
    def single_differential_cumulant4(dcore4): #calculates the differential four particle cummulant, with subevents, returns this and the weight
        #eq 7 in csm paper, equivilent for other POI subevents
        #remove -1234 before the average. 
        for i in range(0, 10, 2):
            mask = (dcore4[i]==-1234)
            dcore4[i] = dcore4[i][~mask]
            dcore4[i+1] = dcore4[i+1][~mask]    
        #mask = (dcor[0]==1234)
        # print(dcor4[1])
        
        A1 = np.average(dcore4[0], weights=dcore4[1])
        A2 = np.average(dcore4[2], weights=dcore4[3])
        A3 = np.average(dcore4[4], weights=dcore4[5])
        A4 = np.average(dcore4[6], weights=dcore4[7])
        A5 = np.average(dcore4[8], weights=dcore4[9])
        
        return A1 - A2*A3 - A4*A5, np.sum(dcore4[1])
    def final_differential_cumulant4(dcor4ae4, dcor4be4, dcor4ce4, dcor4de4):
        #eqivilent eq of eq 4.26 of wang paper for differential cumualnts and 4 instead of 3 subevents
        d_2a, wa = single_differential_cumulant4(dcor4ae4)
        d_2b, wb = single_differential_cumulant4(dcor4be4)
        d_2c, wc = single_differential_cumulant4(dcor4ce4)
        d_2d, wd = single_differential_cumulant4(dcor4de4 )
        return (d_2a*wa+d_2b*wb+d_2c*wc+d_2d*wd)/(wa+wb+wc+wd)

        
    def single_differential_cumulant2(dcore2): #calculates the differential four particle cummulant, with 2 subevents, returns this and the weight
        #eq 4 in csm paper, equivilent for other POI subevents
        #remove -1234 before the average. 
        for i in range(0, 6, 2):
            mask = (dcore2[i]==-1234)
            dcore2[i] = dcore2[i][~mask]
            dcore2[i+1] = dcore2[i+1][~mask]    
        
        A1 = np.average(dcore2[0], weights=dcore2[1])
        A2 = np.average(dcore2[2], weights=dcore2[3])
        A3 = np.average(dcore2[4], weights=dcore2[5])
        return A1 - 2*A2*A3, np.sum(dcore2[1])
    def final_differential_cumulant2(dcor4ae2, dcor4be2):
        #eqivilent eq of eq 4.26 of wang paper for differential cumualnts and 2 instead of 3 subevents
        d_2a, wa = single_differential_cumulant2(dcor4ae2)
        d_2b, wb = single_differential_cumulant2(dcor4be2)
        return (d_2a*wa+d_2b*wb)/(wa+wb)

        
    def cumulants0(dcor, cor):  #calculates the cumulants for 0 subevents based on weights and corelators given. 
        #note dcor is in the shape [dcor4, dcor4w, dcor2 , dcor2w ], similar for cor
        # of sumulant formulas paper
        #dn{2} is given by eq 30
        #dn{4} by eq 34
        # cn{4} by eq 12, cn{2} by eq 11
        for i in range(0, 4, 2):
            mask = (dcor[i]==-1234)
            dcor[i] = dcor[i][~mask]
            dcor[i+1] = dcor[i+1][~mask] 
        #for i in range(0, 4, 2):
            #mask = (cor[i]==-1234)
            cor[i] = cor[i][~mask]
            cor[i+1] = cor[i+1][~mask] 
        dn2 = np.average(dcor[2], weights=dcor[3])
        dn4 = np.average(dcor[0], weights=dcor[1])-2*np.average(dcor[2], weights=dcor[3])*np.average(cor[2], weights= cor[3])

        cn2 = np.average(cor[2], weights=cor[3])
        cn4 = np.average(cor[0], weights=cor[1])-2*np.average(cor[2], weights= cor[3])**2
        return dn4, dn2, cn4, cn2
        

    #this breaks my corelator blocks into 20 separate parts, is hopefully faster than my slicing method
    #before dcor4ae4 was <4>_a', Weights for a, <2>_a',c, Weight, ...
    #now, dcor4ae4_blocks[i] is the ith split of this, so if dcor4ae4 has len 100. and dcor4ae4[0] = [2, 3, 4.5, ..., 4]
    # then dcor4ae4_blocks[0][0] = [2, 3, 4.5, ...] and dcor4ae4_blocks[19][0] = [..., 4]
    n_splits = 20
    dcor4ae4_blocks = [
        [np.array_split(np.asarray(dcor4ae4[i]), n_splits)[k] for i in range(10)]
        for k in range(n_splits)
    ]
    del dcor4ae4
    
    dcor4be4_blocks = [
        [np.array_split(np.asarray(dcor4be4[i]), n_splits)[k] for i in range(10)]
        for k in range(n_splits)
    ]
    del dcor4be4
    
    dcor4ce4_blocks = [
        [np.array_split(np.asarray(dcor4ce4[i]), n_splits)[k] for i in range(10)]
        for k in range(n_splits)
    ]
    del dcor4ce4
    
    dcor4de4_blocks = [
        [np.array_split(np.asarray(dcor4de4[i]), n_splits)[k] for i in range(10)]
        for k in range(n_splits)
    ]
    del dcor4de4
    
    dcore0_blocks = [
        [np.array_split(np.asarray(dcore0[i]), n_splits)[k] for i in range(4)]
        for k in range(n_splits)
    ]
    del dcore0
    
    core0_blocks = [
        [np.array_split(np.asarray(core0[i]), n_splits)[k] for i in range(4)]
        for k in range(n_splits)
    ]
    del core0 

    dcor4ae2_blocks = [
        [np.array_split(np.asarray(dcor4ae2[i]), n_splits)[k] for i in range(6)]
        for k in range(n_splits)
    ]
    del dcor4ae2
    
    dcor4be2_blocks = [
        [np.array_split(np.asarray(dcor4be2[i]), n_splits)[k] for i in range(6)]
        for k in range(n_splits)
    ]
    del dcor4be2
    
    mean_storage_dcor4e4 = []
    mean_storage_dcor4e2 = [ ]
    mean_storage_dcor2e0 = []
    mean_storage_dcor4e0 = []
    mean_storage_cor2e0 = []
    mean_storage_cor4e0 = []
    for i in range(20):
        mean_storage_dcor4e4.append(final_differential_cumulant4(dcor4ae4_blocks[i], dcor4be4_blocks[i], dcor4ce4_blocks[i], dcor4de4_blocks[i]) )
        mean_storage_dcor4e2.append(final_differential_cumulant2(dcor4ae2_blocks[i], dcor4be2_blocks[i]) )
        dn4, dn2, cn4, cn2 = cumulants0(dcore0_blocks[i], core0_blocks[i])
        mean_storage_dcor4e0.append( dn4)
        mean_storage_dcor2e0.append( dn2)
        mean_storage_cor4e0.append( cn4)
        mean_storage_cor2e0.append( cn2)
    return [np.mean(mean_storage_dcor4e4), np.std(mean_storage_dcor4e4), np.mean(mean_storage_dcor4e0), np.std(mean_storage_dcor4e0),
            np.mean(mean_storage_dcor2e0), np.std(mean_storage_dcor2e0), np.mean(mean_storage_cor4e0), np.std(mean_storage_cor4e0), 
            np.mean(mean_storage_cor2e0), np.std(mean_storage_cor2e0), np.mean(mean_storage_dcor4e2), np.std(mean_storage_dcor4e2)] 
    #shape is mean, std, for dn4e4, dn4, dn2, cn4, cn2   

#data theif of MCS plot. 

#note that manual inspection of the points reveals that teh first point is not right for the green triangles. 
import pandas as pd
import matplotlib.pyplot as plt

CMS_data = pd.read_csv("saved_runs/CMS_plot.csv")

fig, ax = plt.subplots()
CMS_nosub = CMS_data['HIJING_wo_subevents_red_filled_circle']*10**(-3) #the 10^-3 is to have the same scale as the plott 
CMS_4sub = CMS_data['HIJING_4subevents_green_filled_triangle']*10**(-3) #the 10^-3 is to have the same scale as the plot
ax.plot(CMS_data['pT_GeV_red'], CMS_nosub,
        'o', color='red', label='HIJING w/o subevents')
ax.plot(CMS_data['pT_GeV_green'], CMS_4sub,
        '^', color='green', label='HIJING 4 subevents')
ax.axhline(0, color='black', linestyle='--')
ax.set_xlabel('$p_T$ (GeV)')
ax.set_ylabel(r'd$_2$\{4\} ')
ax.legend()
plt.show()

#making the data to plot
    
bin_size = 1
n=2
# base_string="nudge_PPb"
end_cols = 11
mult_range = [100, 200]

cn4e0 = np.zeros(100)
cn4stde0 = np.zeros(100)
cn2e0 = np.zeros(100)
cn2stde0 = np.zeros(100) 

dn4e0 = np.zeros(100)
dn4stde0 = np.zeros(100)
dn2e0 = np.zeros(100)
dn2stde0 = np.zeros(100) 

dn4e4 = np.zeros(100)
dn4stde4 = np.zeros(100)

dn4e2 = np.zeros(100)
dn4stde2 = np.zeros(100)

pt_bins = np.zeros(100) 

#CMS_pt = CMS_data['pT_GeV_green'].to_numpy()
# CMS_pt = np.array([ 0.402,  0.713,  1.213,  1.717,  2.219,  2.718,  3.396,  4.414,
#         5.426,  6.766,  8.82 , 10.86] )
CMS_pt = np.array([ 0.402,  0.713,  1.213,  1.717,  2.219,  2.718,  3.396,  4.414] )
for i in range(0, len(CMS_pt)): 
    POI_start = CMS_pt[i]
    if (i==len(CMS_pt)):
        POI_end = 100
    else:
        POI_end = CMS_pt[i+1]

    results = pt_binning(testing12, mult_range, n, POI_start=POI_start, POI_end = POI_end,  momentum_cut=0, nudge=False) 
    print(results)
  
    dn4e4[i] = results[0]
    dn4stde4[i] = results[1]/np.sqrt(20)

    dn4e0[i] = results[2]
    dn4stde0[i] = results[3]/np.sqrt(20)
    dn2e0[i] = results[4]
    dn2stde0[i] = results[5]/np.sqrt(20)
    
    cn4e0[i] = results[6]
    cn4stde0[i] = results[7]/np.sqrt(20)
    cn2e0[i] = results[8]
    cn2stde0[i] = results[9]/np.sqrt(20)

    dn4e2[i] = results[10]
    dn4stde2[i] = results[11]/np.sqrt(20)

    pt_bins[i] = CMS_pt[i]

dn4e4 = np.trim_zeros(dn4e4)
dn4stde4 = np.trim_zeros(dn4stde4)

dn4e2 = np.trim_zeros(dn4e2)
dn4stde2 = np.trim_zeros(dn4stde2)

dn4e0 = np.trim_zeros(dn4e0)
dn4stde0 = np.trim_zeros(dn4stde0)
dn2e0 = np.trim_zeros(dn2e0)
dn2stde0 = np.trim_zeros(dn2stde0)

cn4e0 = np.trim_zeros(cn4e0)
cn4stde0 = np.trim_zeros(cn4stde0)
cn2e0 = np.trim_zeros(cn2e0)
cn2stde0 = np.trim_zeros(cn2stde0)

pt_bins = np.trim_zeros(pt_bins)

arrays_dataframe = {
        "cn4e0": cn4e0,
        "cn4stde0": cn4stde0,
        "cn2e0": cn2e0,
        "cn2stde0": cn2stde0,
        "dn4e0": dn4e0,
        "dn4stde0": dn4stde0,
        "dn2e0": dn2e0,
        "dn2stde0": dn2stde0,
        "dn4e4": dn4e4,
        "dn4stde4": dn4stde4,
        "dn4e2": dn4e2,
        "dn4stde2": dn4stde2,
        "pt_bins": pt_bins
    
    }
arrays_dataframe     


# Create a DataFrame by aligning arrays by index (shorter ones become NaN)
df = pd.DataFrame(dict((k, pd.Series(v)) for k, v in arrays_dataframe.items()))
df.to_csv("sevents0,2,4 100_200_Hijing_anaylsis.csv", index=False)     

#plotting

title_base = 'Hijing, mult=[100-200], p-pb, '
x_label = 'pt'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
cn4e0        = [[], []] 
cn4stde0     = [[], []]
cn2e0        = [[], []]
cn2stde0     = [[], []]
dn4e0        = [[], []] 
dn4stde0     = [[], []]
dn2e0        = [[], []]
dn2stde0     = [[], []] 
dn4e4        = [[], []] 
dn4stde4     = [[], []]


pt = [[], []]
#code that reads in hte saved values, was tested and makes the same plot, in different notebook .
for i in range(1):
    #momentum  = [0.0, 0.7]
    momentum  = [0.0]
    
    #df = pd.read_csv("testing_CSM_ptHijing_anaylsis.csv")
    df = pd.read_csv("100-200CSM_ptHijing_anaylsis.csv")
    
    # Create separate NumPy arrays for each column
    cn4e0[i]        = df["cn4e0"       ].to_numpy()
    cn4stde0[i]     = df["cn4stde0"    ].to_numpy()
    cn2e0[i]       = df["cn2e0"       ].to_numpy()
    cn2stde0[i]     = df["cn2stde0"    ].to_numpy()
    dn4e0[i]        = df["dn4e0"       ].to_numpy()
    dn4stde0[i]     = df["dn4stde0"    ].to_numpy()
    dn2e0[i]       = df["dn2e0"       ].to_numpy()
    dn2stde0[i]     = df["dn2stde0"    ].to_numpy()
    dn4e4[i]        = df["dn4e4"       ].to_numpy()
    dn4stde4[i]     = df["dn4stde4"    ].to_numpy()
    pt[i]  = df["pt_bins"].to_numpy()

    
plt.figure(0)
plt.scatter(pt[0], dn4e4[0],  marker='^',  facecolors='none',edgecolors='green', label = 'calculated $d_2\\{4\\} 4 subevents') #minbias
plt.errorbar(pt[0], dn4e4[0],  yerr=dn4stde4[0] , ecolor='red', linestyle='none')
plt.scatter(CMS_data['pT_GeV_green'], CMS_4sub, marker = '^', color = 'green', label = 'CMS $d_2\\{4\\} 4 subevents') #minbias
# plt.errorbar(pt[0], dn4e4[0],  yerr=dn4stde4[0] , ecolor='red', linestyle='none')
plt.xlim(0, 10)
plt.ylim(-.00025, .0001)


plt.title(title_base+'$d_2\\{4\\}$, 4 subevnts', fontsize = 15)
plt.ylabel("$d_2\\{4\\}$", fontsize = 12) 
plt.xlabel(x_label, fontsize = 12)   
plt.legend()

plt.figure(1)
plt.scatter(pt[0], dn4e0[0],  marker='o',  facecolors='none',edgecolors='red', label = 'calculated $d_2\\{4\\}$ 0 subevents') #minbias
plt.errorbar(pt[0], dn4e0[0],  yerr=dn4stde0[0] , ecolor='red', linestyle='none')
plt.scatter(CMS_data['pT_GeV_red'], CMS_nosub, marker = 'o', color = 'red', label = 'CMS $d_2\\{4\\} 0 subevents') #CMS  
#print(

#old dn4e0 without the weights
# xx = [1,2,3,4,5,6,7]
# yyy = [-4.339150690482422e-05, -4.0682355383022836e-05, -3.4723915297962205e-05, -3.7009292040129205e-05, -2.3049754078975517e-05, -4.469431464210346e-05, -1.174165864141442e-05]
# plt.scatter(xx, yyy,marker='o',  facecolors='none',edgecolors='red', label = 'calculated $d_2\\{4\\}$ 0 subevents, no weights')
# plt.errorbar(pt[0], dn4e4[0],  yerr=dn4stde4[0] , ecolor='red', linestyle='none')
plt.xlim(0, 8)
plt.ylim(-.05*10**(-3), .09*10**(-3))


plt.title(title_base+'$d_2\\{4\\}$, 0 subevnts', fontsize = 15)
plt.ylabel("$d_2\\{4\\}$", fontsize = 12) 
plt.xlabel(x_label, fontsize = 12)   
plt.legend()