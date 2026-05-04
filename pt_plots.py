import pandas as pd
import numpy as np
#first is reading the and doingt he 20 subgroupss 
binsss_size = 10 #what the bin size is
for momentum in np.array([0.0, 1.0, 5.0]): 
    
    # import matplotlib.pyplot as plt 
    
    cn4e0 = np.zeros(100)
    cn4stde0 = np.zeros(100)
    cn2e0 = np.zeros(100)
    cn2stde0 = np.zeros(100) 
    
    cn4e2 = np.zeros(100)
    cn4stde2 = np.zeros(100)
    cn2e2 = np.zeros(100)
    cn2stde2 = np.zeros(100)
    
    cn4e4 = np.zeros(100)
    cn4stde4 = np.zeros(100)
    
    cnac = np.zeros(100)
    cnbd= np.zeros(100)
    cnad= np.zeros(100)
    cnbc= np.zeros(100) 
    multiplicity_bins = np.zeros(100)
    
    
    for i in range(50, 250, binsss_size):
        #mean storeage
        mean_store4e0 = []
        mean_store2e0 = []
        mean_store4e2 = []
        mean_store2e2 = []
        mean_store4e4 = []
        #reading in all the columns
        fucking_columns = str(i)
        cor4e0 = pd.read_csv("ptcor4e0_"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        cor2e0 = pd.read_csv("ptcor2e0_"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        wweight = pd.read_csv("ptwweight"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        mask = (cor4e0 == 0) | ( cor4e0==1234) 
        #apply same masking to all arries, keeps indexs teh same and removes nan values
        cor4e0 = cor4e0[~mask]
        cor2e0 = cor2e0[~mask]
        weight = wweight[~mask] 
        slicee = len(cor4e0) //20 
        # print(slicee)
        for j in range(0, len(cor4e0)-len(cor4e0)%20, slicee):
    
            mean_store4e0.append(np.average(cor4e0[j:j+slicee], weights = weight[j: j+ slicee])-2*np.average(cor2e0[j:j+slicee],weights = weight[j: j+ slicee])**2)
            mean_store2e0.append(np.average(cor2e0[j:j+slicee],weights = weight[j: j+ slicee]))

        
        del cor4e0
        del cor2e0
        cor4e2 = pd.read_csv("ptcor4e2_"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        cor2e2 = pd.read_csv( "ptcor2e2_"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        #wweight = pd.read_csv("ptwweight"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        mask = (cor4e2 == 0) | ( cor4e2==1234)
        cor4e2 = cor4e2[~mask]
        cor2e2 = cor2e2[~mask]
        weight = wweight[~mask] 
        slicee = len(cor4e2) //20
        for j in range(0, len(cor4e2)-len(cor4e2)%20, slicee):
            mean_store4e2.append(np.average(cor4e2[j:j+slicee], weights = weight[j: j+ slicee])-2*np.average(cor2e2[j:j+slicee],weights = weight[j: j+ slicee])**2)
            mean_store2e2.append(np.average(cor2e2[j:j+slicee],weights = weight[j: j+ slicee]))

        
        del cor4e2
        del cor2e2
        cor4e4 = pd.read_csv("ptcor4e4_"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        cnac_idk = pd.read_csv("ptcnac_idk"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        cnbd_idk = pd.read_csv("ptcnbd_idk"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        cnbc_idk = pd.read_csv("ptcnbc_idk"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        cnad_idk = pd.read_csv("ptcnad_idk"+str(momentum)+".csv", usecols = [fucking_columns]).to_numpy().flatten()
        mask = (cor4e4 == 0) | ( cor4e4==1234) 
        #apply same masking to all arries, keeps indexs teh same and removes nan values
         
        cor4e4 = cor4e4[~mask] 
        cnac_idk = cnac_idk[~mask] 
        cnbd_idk = cnbd_idk[~mask]
        cnad_idk = cnad_idk[~mask] 
        cnbc_idk = cnbc_idk[~mask]
        
        weight = wweight[~mask] 
        print(i)
        
        
    
        slicee = len(cor4e4) //20
        # print(slicee)
        for j in range(0, len(cor4e4)-len(cor4e4)%20, slicee):
            mean_store4e4.append(np.average(cor4e4[j:j+slicee],weights = weight[j: j+ slicee])
                               -np.average(cnac_idk[j:j+slicee] , weights = weight[j: j+ slicee])*np.average(cnbd_idk[j:j+slicee], weights = weight[j: j+ slicee])
                              -np.average(cnad_idk[j:j+slicee], weights = weight[j: j+ slicee])*np.average(cnbc_idk[j:j+slicee], weights = weight[j: j+ slicee]))
    
    
        cn4e0[i//binsss_size] = np.array(mean_store4e0).mean()
        cn4stde0[i//binsss_size]=np.array(mean_store4e0).std()/np.sqrt(20)
        cn2e0[i//binsss_size] = np.array(mean_store2e0).mean()
        cn2stde0[i//binsss_size]=np.array(mean_store2e0).std()/np.sqrt(20)
        
        cn4e2[i//binsss_size] = np.array(mean_store4e2).mean()
        cn4stde2[i//binsss_size]=np.array(mean_store4e2).std()/np.sqrt(20)
        cn2e2[i//binsss_size] = np.array(mean_store2e2).mean()
        cn2stde2[i//binsss_size]=np.array(mean_store2e2).std()/np.sqrt(20)
        
        cn4e4[i//binsss_size] = np.array(mean_store4e4).mean()
        cn4stde4[i//binsss_size]=np.array(mean_store4e4).std()/np.sqrt(20)
        
        multiplicity_bins[i//binsss_size] = i
        
    cn4e0 = np.trim_zeros(cn4e0)   
    cn4stde0 = np.trim_zeros(cn4stde0) 
    cn2e0 = np.trim_zeros(cn2e0)   
    cn2stde0 = np.trim_zeros(cn2stde0) 
    
    cn4e2 = np.trim_zeros(cn4e2)   
    cn4stde2 = np.trim_zeros(cn4stde2) 
    cn2e2 = np.trim_zeros(cn2e2)   
    cn2stde2 = np.trim_zeros(cn2stde2 )  
    
    cn4e4 = np.trim_zeros(cn4e4)   
    cn4stde4 = np.trim_zeros(cn4stde4) 
    
    multiplicity_bins = np.trim_zeros(multiplicity_bins)
    arrays_dataframe = {
        "cn4e0": cn4e0,
        "cn4stde0": cn4stde0,
        "cn2e0": cn2e0,
        "cn2stde0": cn2stde0,
        "cn4e2": cn4e2,
        "cn4stde2": cn4stde2,
        "cn2e2": cn2e2,
        "cn2stde2": cn2stde2,
        "cn4e4": cn4e4,
        "cn4stde4": cn4stde4,
        "multiplicity_bins": multiplicity_bins
    
    }
    arrays_dataframe    
    
    
    # Create a DataFrame by aligning arrays by index (shorter ones become NaN)
    df = pd.DataFrame(dict((k, pd.Series(v)) for k, v in arrays_dataframe.items()))
    df.to_csv("ptmin_bias_"+str(momentum)+".csv", index=False)  
    
cn4e0        = [[], []]
cn4stde0     = [[], []]
cn2e0        = [[], []]
cn2stde0     = [[], []]
cn4e2        = [[], []]
cn4stde2     = [[], []]
cn2e2        = [[], []]
cn2stde2     = [[], []]
multiplicity_bins = [[], []]
#code that reads in hte saved values, was tested and makes the same plot, in different notebook .
for i in range(2):
    momentum  = [0.0, 1.0]
    
    df = pd.read_csv("ptmin_bias_"+str(momentum[i])+".csv")
    
    # Create separate NumPy arrays for each column
    cn4e0[i]        = df["cn4e0"       ].to_numpy()
    cn4stde0[i]     = df["cn4stde0"    ].to_numpy()
    cn2e0[i]       = df["cn2e0"       ].to_numpy()
    cn2stde0[i]     = df["cn2stde0"    ].to_numpy()
    cn4e2[i]        = df["cn4e2"       ].to_numpy()
    cn4stde2[i]    = df["cn4stde2"    ].to_numpy() 
    cn2e2[i]        = df["cn2e2"       ].to_numpy()
    cn2stde2[i]     = df["cn2stde2"    ].to_numpy() 
    multiplicity_bins[i]  = df["multiplicity_bins"].to_numpy()
plt.figure(1)
plt.scatter(multiplicity_bins[0], cn4e0[0], color = 'orange', label = 'all events, no subevent') #minbias
plt.errorbar(multiplicity_bins[0], cn4e0[0],  yerr=cn4stde0[0] , ecolor='red', linestyle='none')
plt.scatter(multiplicity_bins[0], cn4e2[0], color = 'blue', label = '2 subevents, everything') #minbias
plt.errorbar(multiplicity_bins[0], cn4e2[0],  yerr=cn4stde2[0] , ecolor='red', linestyle='none')
plt.axhline(0, color='red')
plt.title('1,2 subevents, Min bias, $C_2\{4\}$', fontsize = 15)
plt.ylabel("$C_2\{4\}$", fontsize = 12) 
plt.xlabel('Multiplicty, binned by 10', fontsize = 12)   
plt.legend()
plt.figure(2)
plt.scatter(multiplicity_bins[0], cn2e0[0], color = 'orange', label = 'all events, no subevent') #minbias
plt.errorbar(multiplicity_bins[0], cn2e0[0],  yerr=cn2stde0[0] , ecolor='red', linestyle='none')
plt.scatter(multiplicity_bins[0], cn2e2[0], color = 'blue', label = '2 subevents, everything') #minbias
plt.errorbar(multiplicity_bins[0], cn2e2[0],  yerr=cn2stde2[0] , ecolor='red', linestyle='none')
plt.axhline(0, color='red')
plt.title('1,2 subevents, Min bias, $C_2\{4\}$', fontsize = 15)
plt.ylabel("$C_2\{4\}$", fontsize = 12) 
plt.xlabel('Multiplicty, binned by 10', fontsize = 12)   
plt.legend()

plt.figure(3)
plt.scatter(multiplicity_bins[0], cn4e0[0], color = 'orange', label = 'all events, no subevent') #minbias
plt.errorbar(multiplicity_bins[0], cn4e0[0],  yerr=cn4stde0[0] , ecolor='red', linestyle='none')

plt.scatter(multiplicity_bins[1], cn4e0[1], color = 'blue', label = 'no subevents, pt>1') #minbias
plt.errorbar(multiplicity_bins[1], cn4e0[1],  yerr=cn4stde0[1] , ecolor='red', linestyle='none')

plt.scatter(multiplicity_bins[0], cn4e2[0], color = 'black', label = '2 subevents, everything') #minbias
plt.errorbar(multiplicity_bins[0], cn4e2[0],  yerr=cn4stde2[0] , ecolor='red', linestyle='none')

plt.scatter(multiplicity_bins[1], cn4e2[1], color = 'green', label = '2 subeevents, pt>1') #minbias
plt.errorbar(multiplicity_bins[1], cn4e2[1],  yerr=cn4stde2[1] , ecolor='red', linestyle='none')

plt.axhline(0, color='red')
plt.title('1,2 subevents, Min bias, $C_2\{4\}$', fontsize = 15)
plt.ylabel("$C_2\{4\}$", fontsize = 12) 
plt.xlabel('Multiplicty, binned by 10', fontsize = 12)   
plt.legend() 

plt.figure(4)
plt.scatter(multiplicity_bins[0][5: -1], cn4e0[0][5: -1], color = 'orange', label = 'all events, no subevent') #minbias
plt.errorbar(multiplicity_bins[0][5: -1], cn4e0[0][5: -1],  yerr=cn4stde0[0][5: -1] , ecolor='red', linestyle='none')

plt.scatter(multiplicity_bins[1][5: -1], cn4e0[1][5: -1], color = 'blue', label = 'no subevents, pt>1') #minbias
plt.errorbar(multiplicity_bins[1][5: -1], cn4e0[1][5: -1],  yerr=cn4stde0[1][5: -1] , ecolor='red', linestyle='none')

plt.scatter(multiplicity_bins[0][5: -1], cn4e2[0][5: -1], color = 'black', label = '2 subevents, everything') #minbias
plt.errorbar(multiplicity_bins[0][5: -1], cn4e2[0][5: -1],  yerr=cn4stde2[0][5: -1] , ecolor='red', linestyle='none')

plt.scatter(multiplicity_bins[1][5: -1], cn4e2[1][5: -1], color = 'green', label = '2 subeevents, pt>1') #minbias
plt.errorbar(multiplicity_bins[1][5: -1], cn4e2[1][5: -1],  yerr=cn4stde2[1][5: -1] , ecolor='red', linestyle='none')

plt.axhline(0, color='red')
plt.title('1,2 subevents, Min bias, $C_2\{4\}$', fontsize = 15)
plt.ylabel("$C_2\{4\}$", fontsize = 12) 
plt.xlabel('Multiplicty, binned by 10', fontsize = 12)   
plt.legend()