\begin{lstlisting}[language=Python]
# import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch
import random, numpy as np, pandas as pd
from scipy.optimize import minimize
import scipy.cluster
import math
import statistics
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Equally Weighted Portfolio
def ewPortfolio(cov,**kargs):
    n=len(cov)
    return n*[1/n]
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Risk Parity
def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = np.dot(V,w.T)
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T)) # sum of squared error
    return J

def risk_parity(cov,**kargs):
    riskbudget=np.array([1/len(cov)]*len(cov))
    w0=np.array([1/len(cov)]*len(cov))
    x_t = riskbudget # your risk budget percent of total portfolio risk (equal risk)
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
    {'type': 'ineq', 'fun': long_only_constraint})
    res= minimize(risk_budget_objective, w0, args=[cov,x_t],tol=0.000000000000001,
                  method='SLSQP',constraints=cons) 
    return res.x
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Global minimum variance - long only
def GMVLOPortfolio(cov,**kargs):
    x0=pd.Series([1/len(cov)]*len(cov))
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
    {'type': 'ineq', 'fun': long_only_constraint})
    res = minimize(calculate_portfolio_var,x0,args=cov,tol=0.000000000000001,
                   method='SLSQP',constraints=cons)
    return res.x
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Inverse variance portfolio (Risk parity fra De Prado)
def getIVP(cov,**kargs):
    #Compute the inverse-variance portfolio
    ivp=1/np.diag(cov)
    ivp/=ivp.sum()
    return ivp
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Maximum diversification
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    return (np.dot(np.dot(w,V),w.T))

def calc_diversification_ratio(w, V):
    # average weighted vol
    w_vol = np.dot(np.sqrt(np.diag(V)), w.T)
    # portfolio vol
    port_vol = np.sqrt(calculate_portfolio_var(w, V))
    diversification_ratio = w_vol/port_vol
    # return negative for minimization problem (maximize = minimize -)
    return -diversification_ratio

def total_weight_constraint(x):
    return np.sum(x)-1

def long_only_constraint(x):
    return x

def max_div_port(cov,**kargs):
    # w0: initial weight
    # V: covariance matrix
    # bnd: individual position limit
    # long only: long only constraint
    bnd=None
    long_only=True
    w0=np.array([1/len(cov)]*len(cov))
    cons = ({'type': 'eq', 'fun': total_weight_constraint},)
    if long_only: # add in long only constraint
        cons = cons + ({'type': 'ineq', 'fun':  long_only_constraint},)
    res = minimize(calc_diversification_ratio, w0, bounds=bnd,
                   args=cov, method='SLSQP', constraints=cons)
    return res.x
\end{lstlisting}

\begin{lstlisting}[language=Python]
#HRP2
#This version of HRP divides the weights between clusters

import collections
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
    
def get_cluster_dict(link):
    link=np.append(link,np.array([[j] for j in range(int(link[-1,3]),
                                                     int(link[-1,3])*2-1)]),axis=1)
    cluster_dict={}
    for i in link[:,0:5].astype(int):
        cluster_dict[i[4]]=[]
        if i[0]>=link[0,-1]:
            cluster_dict[i[4]].append(cluster_dict[i[0]])
        else:
            cluster_dict[i[4]].append(i[0])
        if i[1]>=link[0,-1]:
            cluster_dict[i[4]].append(cluster_dict[i[1]])
        else:
            cluster_dict[i[4]].append(i[1])
        
    return cluster_dict


def recClusterVar(cluster_dict,link, cov):
    link=np.append(link,np.array([[j] for j in range(int(link[-1,3]),
                                                     int(link[-1,3])*2-1)]),axis=1)
    w=pd.Series(1,index=[i for i in range(int(link[0,-1]))]) 
    for i in reversed(link.astype(int)):
        if i[0]>=link[0,-1]:
            cluster1 = cluster_dict[i[0]]
        else:
            cluster1 = i[0]

        if i[1]>=link[0,-1]:
            cluster2 = cluster_dict[i[1]]
        else:
            cluster2 = i[1]
        cluster1=[i for i in flatten(cluster1)]
        cluster2=[i for i in flatten(cluster2)]
        c1_var=getClusterVar(cov,cluster1)
        c2_var=getClusterVar(cov,cluster2)
        alpha=1-c1_var/(c1_var+c2_var)
        w[cluster1]*=alpha # weight 1
        w[cluster2]*=1-alpha # weight 2
    return w
\end{lstlisting}

\begin{lstlisting}[language=Python]
#HRP
def getClusterVar(cov,cItems):
    #Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] #number of original items
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) #make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i = df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx=sortIx.append(df0) # item 2
        sortIx=sortIx.sort_index() #re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()

def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,
                k in ((0,len(i)//2),(len(i)//2,len(i))) if len(i)>1] # bi-section
        
        for i in range(0,len(cItems),2):
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
            
    w.sort_index(inplace=True)
    return w
    

def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    #This is a proper diastance metric
    dist=((1-corr)/2.)**.5 # distance matrix
    return dist


def plotCorrMatrix(path,corr,labels=None):
    #Heatmap of the correlation matrix
    if labels is None: labels=[]
    mpl.pcolor(corr)
    mpl.colorbar()
    mpl.yticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.xticks(np.arange(.5,corr.shape[0]+.5),labels)
    mpl.savefig(path)
    mpl.clf();mpl.close()  #reset pylab
    return

    
def findCorrelatedCols(colnbs,size0):
    keys = list(set([i[0] for i in colnbs]))
    for i in range(1,size0+1):
        if i not in keys:
            keys.append(i)
    keys.sort()
    clusters={key: [key] for key in keys}
    for i in colnbs:
        clusters[i[0]].append(i[1])
    return clusters

def clusterWeights(clusters, hrp):
    weights={key:None for key in clusters.keys()}
    for i in weights:
        weights[i] = sum([hrp.loc[j] for j in clusters[i]])
    return list(weights.values())
    
    
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Barplots of allocations
def plot_weights(hrp, hrp2, cov, data):
    index = list(data.columns)
    
    #HRP
    hrp = hrp.sort_values(ascending=False)
    hrp.plot.bar(figsize = (15,7))
    mpl.title("HRP")
    mpl.ylabel("Weight")
    mpl.xlabel("Asset")
    mpl.ylim((0, 0.5))
    mpl.show()
    
    #HRP2
    hrp2 = hrp2.sort_values(ascending=False)
    hrp2.plot.bar(figsize = (15,7))
    mpl.title("HRP2")
    mpl.ylabel("Weight")
    mpl.xlabel("Asset")
    mpl.ylim((0, 0.5))
    mpl.show()

    #Naive Risk-Parity
    ivp = getIVP(cov)
    ivp = pd.Series(ivp, index=index)
    ivp = ivp.sort_values(ascending=False)
    ivp.plot.bar(figsize = (15,7))
    mpl.title("Naive Risk parity (IVP)")
    mpl.ylabel("Weight")
    mpl.xlabel("Asset")
    mpl.ylim((0,0.5))
    mpl.show()
    
    #Risk Parity
    rp = risk_parity(cov)
    rp = pd.Series(rp, index=index)
    rp = rp.sort_values(ascending=False)
    rp.plot.bar(figsize = (15,7))
    mpl.title("Risk Parity")
    mpl.ylabel("Weight")
    mpl.xlabel("Asset")
    mpl.ylim((0, 0.5))
    mpl.show()
    
    #GMV
    gmv = GMVPortfolio(cov)
    gmv = pd.Series(gmv, index=index)
    gmv = gmv.sort_values(ascending=False)
    gmv.plot.bar(figsize = (15,7))
    mpl.title("GMV")
    mpl.ylabel("Weight")
    mpl.xlabel("Asset")
    mpl.ylim((-0.2, 0.5))
    mpl.show()
    
    #GMV Long-only
    gmvlo = GMVLOPortfolio(cov)
    gmvlo = pd.Series(gmvlo, index=index)
    gmvlo = gmvlo.sort_values(ascending = False)
    gmvlo.plot.bar(figsize=(15,7))
    mpl.title("GMV Long-only")
    mpl.ylabel("weight")
    mpl.xlabel("Asset")
    mpl.ylim((0,0.5))
    mpl.show()
    
    
    #Maximum_Div_port
    mdv = pd.Series(max_div_port(cov), index=index)
    mdv = mdv.sort_values(ascending=False)
    mdv.plot.bar(figsize = (15,7))
    mpl.title("Maximum Diversification Portfolio")
    mpl.ylabel("Weight")
    mpl.xlabel("Asset")
    mpl.ylim((-0.1, 0.5))
    mpl.show()
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Imports dates for the time-series
dates = pd.read_excel('Betting Against Beta Equity Factors Daily.xlsx',
                      'MKT',skiprows=18434,
                      usecols='A', header = 0)
dates.columns=['Date']
dates=pd.Series(dates['Date'])
dd= pd.to_datetime(dates[520:]).reset_index(drop=True)

risk_free_rates = pd.read_excel('Betting Against Beta Equity Factors Daily.xlsx',
                                'RF',skiprows=19001, usecols='B')
risk_free_rates = (risk_free_rates[520:-18])
#Imports all countries in the MKT factor
mkt_data = pd.read_excel('Betting Against Beta Equity Factors Daily.xlsx','MKT',
                         skiprows=18434, usecols='B:Y')
mkt_global = pd.read_excel('Betting Against Beta Equity Factors Daily.xlsx','MKT',
                           skiprows=18434, usecols='Z',
                           header=0)
mkt_data.columns = ['AUS', 'AUT', 'BEL', 'CAN', 'CHE', 'DEU', 'DNK', 'ESP', 'FIN',
                    'FRA', 'GBR', 'GRC', 'HKG', 'IRL', 'ISR', 'ITA', 'JPN', 'NLD', 
                    'NOR', 'NZL', 'PRT', 'SGP', 'SWE', 'USA']
\end{lstlisting}

\begin{lstlisting}[language=Python]
arp_data = mkt_data
\end{lstlisting}

\begin{lstlisting}[language=Python]
def main_3():
    #Back-test on real-world data
    re_calc_time,sample_size = 22,520
    x = pd.DataFrame(arp_data)
    #New
    HRP_portfolio_return=[]
    RP_portfolio_return=[]
    RiskP_portfolio_return=[]
    GMV_portfolio_return=[]
    GMVLO_portfolio_return=[]
    MD_portfolio_return=[]
    EW_portfolio_return=[]
    HRP2_portfolio_return=[]
    
    hrp_weights=[]
    hrp2_weights=[]
    gmv_weights=[]
    mdv_weights=[]
    ivp_weights=[]
    rp_weights=[]
    gmvlo_weights=[]
    
    realisedRC_RP = []
    realisedRC_HRP = []
    realisedRC_HRP2 = []
    realisedRC_GMV = []
    realisedRC_MD = []
    realisedRC_IVP = []
    realisedRC_GMVLO = []
    realisedRC_EW = []
    
    realisedDR_RP = []
    realisedDR_HRP = []
    realisedDR_HRP2 = []
    realisedDR_GMV = []
    realisedDR_MD = []
    realisedDR_IVP = []
    realisedDR_GMVLO = []
    realisedDR_EW = []
    
    EW_weight=[1/len(arp_data.columns)]*len(arp_data.columns)
    returns=[]
    pointers = range(520,len(x)-22,re_calc_time)
    for pointer in pointers:
        #Gets data_sample
        x_sample = x.iloc[pointer-sample_size:pointer] 
        cov,corr=x_sample.cov().reset_index(drop=True),
                    x_sample.corr().reset_index(drop=True)
        cov.columns, corr.columns = [i for i in range(len(cov))],
                                    [i for i in range(len(cov))]
        #HRP
        dist=correlDist(corr)
        link=sch.linkage(dist,'single')
        sortIx=getQuasiDiag(link)
        sortIx=corr.index[sortIx].tolist() #recover labels
        df0=corr.loc[sortIx,sortIx]
        plotCorrMatrix('HRP3_corr{}.png'.format(i),df0,labels=df0.columns)
        hrp=getRecBipart(cov,sortIx).sort_index()
        #HRP2
        cluster_dict = get_cluster_dict(link)
        hrp2 = recClusterVar(cluster_dict,link, cov)#.sort_index()
        #IVP
        ivp = getIVP(cov)
        #GMV
        gmv = GMVPortfolio(cov)
        #GMV Long-only
        gmvlo = GMVLOPortfolio(cov)
        #MD
        mdv = pd.Series(max_div_port(cov))
        #Risk Parity
        rp = pd.Series(risk_parity(cov))
        hrp_weights.append(hrp)
        #Weights
        hrp2_weights.append(hrp2)
        gmv_weights.append(gmv)
        gmvlo_weights.append(gmvlo)
        mdv_weights.append(mdv)
        ivp_weights.append(ivp)
        rp_weights.append(rp)
        
        #realised risk contribution
        cov_out = x.iloc[pointer:pointer+re_calc_time].cov()
        realisedRC_RP.append(calculate_risk_contribution(rp, cov_out)/
                             np.sqrt(calculate_portfolio_var(rp,cov_out)))
        realisedRC_HRP.append(calculate_risk_contribution(hrp, cov_out)/
                              np.sqrt(calculate_portfolio_var(hrp,cov_out)))
        realisedRC_HRP2.append(calculate_risk_contribution(hrp2, cov_out)/
                               np.sqrt(calculate_portfolio_var(hrp2,cov_out)))
        realisedRC_GMV.append(calculate_risk_contribution(gmv, cov_out)/
                              np.sqrt(calculate_portfolio_var(gmv,cov_out)))
        realisedRC_MD.append(calculate_risk_contribution(mdv, cov_out)/
                             np.sqrt(calculate_portfolio_var(mdv,cov_out)))
        realisedRC_IVP.append(calculate_risk_contribution(ivp, cov_out)/
                              np.sqrt(calculate_portfolio_var(ivp,cov_out)))
        realisedRC_GMVLO.append(calculate_risk_contribution(gmvlo, cov_out)/
                                np.sqrt(calculate_portfolio_var(gmvlo,cov_out)))
        realisedRC_EW.append(calculate_risk_contribution(EW_weight, cov_out)/
                             np.sqrt(calculate_portfolio_var(pd.Series(EW_weight),
                                                             cov_out)))
        
        
        #realised diversification ratio
        realisedDR_RP.append(calc_diversification_ratio(rp, cov_out))
        realisedDR_HRP.append(calc_diversification_ratio(hrp, cov_out))
        realisedDR_HRP2.append(calc_diversification_ratio(hrp2, cov_out))
        realisedDR_GMV.append(calc_diversification_ratio(gmv, cov_out))
        realisedDR_MD.append(calc_diversification_ratio(mdv, cov_out))
        realisedDR_IVP.append(calc_diversification_ratio(ivp, cov_out))
        realisedDR_GMVLO.append(calc_diversification_ratio(gmvlo, cov_out))
        realisedDR_EW.append(calc_diversification_ratio(pd.Series(EW_weight),
                                                        cov_out))
        
        for j in range(re_calc_time):
            HRP_portfolio_return.append(np.dot(hrp,arp_data.iloc[pointer+j]))
            RP_portfolio_return.append(np.dot(ivp,arp_data.iloc[pointer+j]))
            GMV_portfolio_return.append(np.dot(gmv,arp_data.iloc[pointer+j]))
            GMVLO_portfolio_return.append(np.dot(gmvlo,arp_data.iloc[pointer+j]))
            MD_portfolio_return.append(np.dot(mdv,arp_data.iloc[pointer+j]))
            EW_portfolio_return.append(np.dot(EW_weight,arp_data.iloc[pointer+j]))
            HRP2_portfolio_return.append(np.dot(hrp2,arp_data.iloc[pointer+j]))
            RiskP_portfolio_return.append(np.dot(rp,arp_data.iloc[pointer+j]))
    return (pd.DataFrame(HRP_portfolio_return), pd.DataFrame(RP_portfolio_return), 
            pd.DataFrame(GMV_portfolio_return), pd.DataFrame(GMVLO_portfolio_return), 
            pd.DataFrame(MD_portfolio_return),pd.DataFrame(EW_portfolio_return), 
            pd.DataFrame(HRP2_portfolio_return), pd.DataFrame(RiskP_portfolio_return), 
            pd.DataFrame(returns),pd.DataFrame(hrp_weights),pd.DataFrame(hrp2_weights),
            pd.DataFrame(gmv_weights),
            pd.DataFrame(mdv_weights),pd.DataFrame(ivp_weights),
            pd.DataFrame(rp_weights), pd.DataFrame(gmvlo_weights),
            realisedRC_RP, realisedRC_HRP, 
            realisedRC_HRP2, realisedRC_GMV, realisedRC_GMVLO,
            realisedRC_IVP, realisedRC_MD, 
            realisedRC_EW, realisedDR_GMV,realisedDR_GMVLO,
             realisedDR_HRP,realisedDR_HRP2,realisedDR_IVP,
            realisedDR_MD,realisedDR_RP,realisedDR_EW)

\end{lstlisting}

\begin{lstlisting}[language=Python]
def value_at_risk(returns, confidence_level=.05):
    return returns.quantile(confidence_level, interpolation='higher')


def expected_shortfall(returns, confidence_level=.05):
    var = value_at_risk(returns, confidence_level)

    return returns[returns.lt(var)].mean()
\end{lstlisting}

\begin{lstlisting}[language=Python]
(HRP_weights,RP_weights,GMV_weights,GMVLO_weights,
 MD_weights,EW_weights,HRP2_weights, RiskP_weights,
 returns, hrp_weights, hrp2_weights, gmv_weights, 
 mdv_weights, ivp_weights, rp_weights, gmvlo_weights, 
 realisedRC_RP, realisedRC_HRP, realisedRC_HRP2, 
 realisedRC_GMV, realisedRC_GMVLO, realisedRC_IVP, 
 realisedRC_MD, realisedRC_EW, realisedDR_GMV, 
 realisedDR_GMVLO, realisedDR_HRP, realisedDR_HRP2, 
 realisedDR_IVP, realisedDR_MD, realisedDR_RP, realisedDR_EW) = main_3()
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Calculating the cummulative returns
HRP_portfolio_returns = HRP_weights
HRP_cumulative_returns = (HRP_portfolio_returns + 1).cumprod()
HRP_cumulative_returns['dates']=dd
HRP_cumulative_returns.set_index('dates',inplace=True,drop=True)

HRP2_portfolio_returns = HRP2_weights
HRP2_cumulative_returns = (HRP2_portfolio_returns +1).cumprod()
HRP2_cumulative_returns['dates']=dd
HRP2_cumulative_returns.set_index('dates',inplace=True,drop=True)

RP_portfolio_returns = RP_weights
RP_cumulative_returns = (RP_portfolio_returns + 1).cumprod()
RP_cumulative_returns['dates']=dd
RP_cumulative_returns.set_index('dates',inplace=True,drop=True)

RiskP_portfolio_returns = RiskP_weights
RiskP_cumulative_returns = (RiskP_portfolio_returns + 1).cumprod()
RiskP_cumulative_returns['dates']=dd
RiskP_cumulative_returns.set_index('dates',inplace=True,drop=True)

GMV_portfolio_returns = GMV_weights
GMV_cumulative_returns = (GMV_portfolio_returns + 1).cumprod()
GMV_cumulative_returns['dates']=dd
GMV_cumulative_returns.set_index('dates',inplace=True,drop=True)

GMVLO_portfolio_returns = GMVLO_weights
GMVLO_cumulative_returns = (GMVLO_portfolio_returns + 1).cumprod()
GMVLO_cumulative_returns['dates']=dd
GMVLO_cumulative_returns.set_index('dates',inplace=True,drop=True)

MD_portfolio_returns = MD_weights
MD_cumulative_returns = (MD_portfolio_returns + 1).cumprod()
MD_cumulative_returns['dates']=dd
MD_cumulative_returns.set_index('dates',inplace=True,drop=True)

EW_portfolio_returns = EW_weights
EW_cumulative_returns = (EW_portfolio_returns + 1).cumprod()
EW_cumulative_returns['dates']=dd
EW_cumulative_returns.set_index('dates',inplace=True,drop=True)

mkt_global_cumulative_returns = (mkt_global[520:] + 1).cumprod()
mkt_global_cumulative_returns =mkt_global_cumulative_returns.reset_index(drop=True)
mkt_global_cumulative_returns['dates']=dd
mkt_global_cumulative_returns.set_index('dates',inplace=True,drop=True)
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Plot cummulative returns of strategies
ax = HRP_cumulative_returns.plot(figsize=[15,10])
RP_cumulative_returns.plot(figsize=[15,10], ax=ax)
RiskP_cumulative_returns.plot(figsize=[15,10],ax=ax)
GMVLO_cumulative_returns.plot(figsize=[15,10], ax=ax)
MD_cumulative_returns.plot(figsize=[15,10], ax=ax)
EW_cumulative_returns.plot(figsize=[15,10], ax=ax)
HRP2_cumulative_returns.plot(figsize=[15,10], ax=ax)

mkt_global_cumulative_returns.plot(figsize=[15,10], ax=ax)

mpl.legend(['HRP', 'Naive RP','RP', 'GMV', 'MD','EW','HRP2', 'Global'])
mpl.ylabel("Accumulated returns (Index 1)")
mpl.yscale("log")
mpl.show()
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Creates table to show the results
l=len(HRP_cumulative_returns)-1
allo_comp = [[(HRP_cumulative_returns[0][-1])**(1/23)-1,
              (HRP2_cumulative_returns[0][-1])**(1/23)-1, 
              (RP_cumulative_returns[0][-1])**(1/23)-1,
              (GMV_cumulative_returns[0][-1])**(1/23)-1, 
              (GMVLO_cumulative_returns[0][-1])**(1/23)-1,
              (MD_cumulative_returns[0][-1])**(1/23)-1,
              (RiskP_cumulative_returns[0][-1])**(1/23)-1 ,
              (EW_cumulative_returns[0][-1])**(1/23)-1]]
allo_comp.append([HRP_portfolio_returns.std()[0]*np.sqrt(260),
                  HRP2_portfolio_returns.std()[0]*np.sqrt(260),
                  RP_portfolio_returns.std()[0]*np.sqrt(260), 
                  GMV_portfolio_returns.std()[0]*np.sqrt(260), 
                  GMVLO_portfolio_returns.std()[0]*np.sqrt(260),
                  MD_portfolio_returns.std()[0]*np.sqrt(260),
                  RiskP_portfolio_returns.std()[0]*np.sqrt(260),
                  EW_portfolio_returns.std()[0]*np.sqrt(260)])
allo_comp.append([expected_shortfall(HRP_portfolio_returns)[0],
                  expected_shortfall(HRP2_portfolio_returns)[0],
                  expected_shortfall(RP_portfolio_returns)[0], 
                  expected_shortfall(GMV_portfolio_returns)[0], 
                  expected_shortfall(GMVLO_portfolio_returns)[0], 
                  expected_shortfall(MD_portfolio_returns)[0],
                  expected_shortfall(RiskP_portfolio_returns)[0], 
                  expected_shortfall(EW_portfolio_returns)[0]])
\end{lstlisting}

\begin{lstlisting}[language=Python]
allo_comp=pd.DataFrame(allo_comp, columns=['HRP', 'HRP2','Naive RP','GMV','GMVLO','MD','Risk Parity','EW'],
                       index=['Annualized return','Annualized standard deviance','Expected shortfall'])
#Adds Sharpe Ratio to the table
allo_comp.loc['Sharpe Ratio']=allo_comp.loc['Annualized return']/allo_comp.loc['Annualized standard deviance']
#Expected shortfall N days
N = 22

HRP_portfolio_returns14 = HRP_portfolio_returns.groupby(HRP_portfolio_returns.index // N).sum()
HRP2_portfolio_returns14 = HRP2_portfolio_returns.groupby(HRP2_portfolio_returns.index // N).sum()
RP_portfolio_returns14 = RP_portfolio_returns.groupby(RP_portfolio_returns.index // N).sum()
GMV_portfolio_returns14 = GMV_portfolio_returns.groupby(GMV_portfolio_returns.index // N).sum()
GMVLO_portfolio_returns14 = GMVLO_portfolio_returns.groupby(GMVLO_portfolio_returns.index // N).sum()
MD_portfolio_returns14 = MD_portfolio_returns.groupby(MD_portfolio_returns.index // N).sum()
RiskP_portfolio_returns14 = RiskP_portfolio_returns.groupby(RiskP_portfolio_returns.index // N).sum()
EW_portfolio_returns14 = EW_portfolio_returns.groupby(EW_portfolio_returns.index // N).sum()
allo_comp.loc['Expected shortfall 14 days'] = [expected_shortfall(HRP_portfolio_returns14)[0],
                                               expected_shortfall(HRP2_portfolio_returns14)[0],
                                               expected_shortfall(RP_portfolio_returns14)[0],
                                               expected_shortfall(GMV_portfolio_returns14)[0],
                                               expected_shortfall(GMVLO_portfolio_returns14)[0],
                                               expected_shortfall(MD_portfolio_returns14)[0],
                                               expected_shortfall(RiskP_portfolio_returns14)[0],
                                               expected_shortfall(EW_portfolio_returns14)[0]]
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Adds realised risk contribution and diversification ratio to the table
allo_comp.loc['Realised deviance risk contribution'] = [portfolio_risk_deviance(realisedRC_HRP),
                                                        portfolio_risk_deviance(realisedRC_HRP2),
                                                        portfolio_risk_deviance(realisedRC_IVP),
                                                        portfolio_risk_deviance(realisedRC_GMV),
                                                        portfolio_risk_deviance(realisedRC_GMVLO),
                                                        portfolio_risk_deviance(realisedRC_MD),
                                                        portfolio_risk_deviance(realisedRC_RP),
                                                        portfolio_risk_deviance(realisedRC_EW)]
allo_comp.loc['Realised diversification ratio'] = [abs(statistics.mean(realisedDR_HRP)),
                                                   abs(statistics.mean(realisedDR_HRP2)),
                                                   abs(statistics.mean(realisedDR_IVP)),
                                                   abs(statistics.mean(realisedDR_GMV)),
                                                   abs(statistics.mean(realisedDR_GMVLO)),
                                                   abs(statistics.mean(realisedDR_MD)),
                                                   abs(statistics.mean(realisedDR_RP)),
                                                   abs(statistics.mean(realisedDR_EW))]
allo_comp
\end{lstlisting}

\begin{lstlisting}[language=Python]
#step 1
returns22 += 1
x_hat_hrp = pd.DataFrame(returns22[25:].values*hrp_weights.values,
                         columns=returns22.columns, index=hrp_weights.index)
x_hat_hrp2 = pd.DataFrame(returns22[25:].values*hrp2_weights.values,
                          columns=returns22.columns, index=hrp2_weights.index)
x_hat_ivp = pd.DataFrame(returns22[25:].values*ivp_weights.values,
                         columns=returns22.columns, index=ivp_weights.index)
x_hat_gmv = pd.DataFrame(returns22[25:].values*gmvlo_weights.values,
                         columns=returns22.columns, index=gmvlo_weights.index)
x_hat_md = pd.DataFrame(returns22[25:].values*mdv_weights.values,
                        columns=returns22.columns, index=mdv_weights.index)
x_hat_rp = pd.DataFrame(returns22[25:].values*rp_weights.values,
                        columns=returns22.columns, index=rp_weights.index)

ew_weights = pd.DataFrame(1/24,index=hrp_weights.index, columns=hrp_weights.columns)
x_hat_ew = pd.DataFrame(returns22[25:].values*ew_weights.values,
                        columns=returns22.columns, index=ew_weights.index)

#step 2
q_hrp = x_hat_hrp.sum(axis=1)
x_tilde_hrp = x_hat_hrp.div(q_hrp, axis=0)

q_hrp2 = x_hat_hrp2.sum(axis=1)
x_tilde_hrp2 = x_hat_hrp2.div(q_hrp2, axis=0)

q_ivp = x_hat_ivp.sum(axis=1)
x_tilde_ivp = x_hat_ivp.div(q_ivp, axis=0)

q_gmv = x_hat_gmv.sum(axis=1)
x_tilde_gmv = x_hat_gmv.div(q_gmv, axis=0)

q_md = x_hat_md.sum(axis=1)
x_tilde_md = x_hat_md.div(q_md, axis=0)

q_rp = x_hat_rp.sum(axis=1)
x_tilde_rp = x_hat_rp.div(q_rp, axis=0)

q_ew = x_hat_ew.sum(axis=1)
x_tilde_ew = x_hat_ew.div(q_ew, axis=0)


#step 3
HRP_Turnover = pd.DataFrame(abs(x_tilde_hrp.iloc[:272].values - hrp_weights.iloc[1:].values),
                            columns=returns22.columns, index=x_tilde_hrp.iloc[:272].index)
HRP_Turnover = HRP_Turnover.sum(axis=1)

HRP2_Turnover = pd.DataFrame(abs(x_tilde_hrp2.iloc[:272].values - hrp2_weights.iloc[1:].values),
                             columns=returns22.columns, index=x_tilde_hrp2.iloc[:272].index)
HRP2_Turnover = HRP2_Turnover.sum(axis=1)

IVP_Turnover = pd.DataFrame(abs(x_tilde_ivp.iloc[:272].values - ivp_weights.iloc[1:].values),
                            columns=returns22.columns, index=x_tilde_ivp.iloc[:272].index)
IVP_Turnover = IVP_Turnover.sum(axis=1)

GMV_Turnover = pd.DataFrame(abs(x_tilde_gmv.iloc[:272].values - gmvlo_weights.iloc[1:].values),
                            columns=returns22.columns, index=x_tilde_gmv.iloc[:272].index)
GMV_Turnover = GMV_Turnover.sum(axis=1)

MD_Turnover = pd.DataFrame(abs(x_tilde_md.iloc[:272].values - mdv_weights.iloc[1:].values),
                           columns=returns22.columns, index=x_tilde_md.iloc[:272].index)
MD_Turnover = MD_Turnover.sum(axis=1)

RP_Turnover = pd.DataFrame(abs(x_tilde_rp.iloc[:272].values - rp_weights.iloc[1:].values),
                           columns=returns22.columns, index=x_tilde_rp.iloc[:272].index)
RP_Turnover = RP_Turnover.sum(axis=1)

EW_Turnover = pd.DataFrame(abs(x_tilde_ew.iloc[:272].values - ew_weights.iloc[1:].values),
                           columns=returns22.columns, index=x_tilde_ew.iloc[:272].index)
EW_Turnover = EW_Turnover.sum(axis=1)

#Transaction cost
HRP_xi = HRP_cumulative_returns.iloc[22::22].mul(0.001)
HRP2_xi = HRP2_cumulative_returns.iloc[22::22].mul(0.001)
IVP_xi = RP_cumulative_returns.iloc[22::22].mul(0.001)
GMV_xi = GMV_cumulative_returns.iloc[22::22].mul(0.001)
MD_xi = MD_cumulative_returns.iloc[22::22].mul(0.001)
RP_xi = RiskP_cumulative_returns.iloc[22::22].mul(0.001)
EW_xi = EW_cumulative_returns.iloc[22::22].mul(0.001)

def flatlist(l):
    flat_list = []
    for sublist in l.values.tolist():
        for item in sublist:
            flat_list.append(item)
    return pd.Series(flat_list)

HRP_xi = flatlist(HRP_xi)
HRP_TC = pd.DataFrame((HRP_Turnover.values*HRP_xi.values))

HRP2_xi = flatlist(HRP2_xi)
HRP2_TC = pd.DataFrame((HRP2_Turnover.values*HRP2_xi.values))

IVP_xi = flatlist(IVP_xi)
IVP_TC = pd.DataFrame((IVP_Turnover.values*IVP_xi.values))

GMV_xi = flatlist(GMV_xi)
GMV_TC = pd.DataFrame((GMV_Turnover.values*GMV_xi.values))

MD_xi = flatlist(MD_xi)
MD_TC = pd.DataFrame((MD_Turnover.values*MD_xi.values))

RP_xi = flatlist(RP_xi)
RP_TC = pd.DataFrame((RP_Turnover.values*RP_xi.values))

EW_xi = flatlist(EW_xi)
EW_TC = pd.DataFrame((EW_Turnover.values*EW_xi.values))#Turnover and transaction costs
returns22 = arp_data.groupby(arp_data.index // 22).sum()
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Subtracting transaction cost every month
HRP_returns_TC = HRP_portfolio_returns
HRP2_returns_TC = HRP2_portfolio_returns
IVP_returns_TC = RP_portfolio_returns
GMV_returns_TC = GMVLO_portfolio_returns
MD_returns_TC = MD_portfolio_returns
RP_returns_TC = RiskP_portfolio_returns

i=0
k=0

while (k<272):
    HRP_returns_TC.iloc[i] = HRP_portfolio_returns.iloc[i].sub(HRP_TC.iloc[k])
    HRP2_returns_TC.iloc[i] = HRP2_portfolio_returns.iloc[i].sub(HRP2_TC.iloc[k])
    IVP_returns_TC.iloc[i] = RP_portfolio_returns.iloc[i].sub(IVP_TC.iloc[k])
    GMV_returns_TC.iloc[i] = GMVLO_portfolio_returns.iloc[i].sub(GMV_TC.iloc[k])
    MD_returns_TC.iloc[i] = MD_portfolio_returns.iloc[i].sub(MD_TC.iloc[k])
    RP_returns_TC.iloc[i] = RiskP_portfolio_returns.iloc[i].sub(RP_TC.iloc[k])
    i += 22
    k += 1
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Cumulative returns with transaction cost
HRP_cumulative_returns_TC = (HRP_returns_TC + 1).cumprod()
HRP2_cumulative_returns_TC = (HRP2_returns_TC + 1).cumprod()
IVP_cumulative_returns_TC = (IVP_returns_TC + 1).cumprod()
GMV_cumulative_returns_TC = (GMV_returns_TC + 1).cumprod()
MD_cumulative_returns_TC = (MD_returns_TC + 1).cumprod()
RP_cumulative_returns_TC = (RP_returns_TC + 1).cumprod()

#Date
HRP_cumulative_returns_TC['dates']=dd
HRP_cumulative_returns_TC.set_index('dates',inplace=True,drop=True)

HRP2_cumulative_returns_TC['dates']=dd
HRP2_cumulative_returns_TC.set_index('dates',inplace=True,drop=True)

IVP_cumulative_returns_TC['dates']=dd
IVP_cumulative_returns_TC.set_index('dates',inplace=True,drop=True)

GMV_cumulative_returns_TC['dates']=dd
GMV_cumulative_returns_TC.set_index('dates',inplace=True,drop=True)

MD_cumulative_returns_TC['dates']=dd
MD_cumulative_returns_TC.set_index('dates',inplace=True,drop=True)

RP_cumulative_returns_TC['dates']=dd
RP_cumulative_returns_TC.set_index('dates',inplace=True,drop=True)
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Plot of cumulative returns with transaction costs
ax = HRP_cumulative_returns_TC.plot(figsize=[15,10])
IVP_cumulative_returns_TC.plot(figsize=[15,10], ax=ax)
RP_cumulative_returns_TC.plot(figsize=[15,10],ax=ax)
GMV_cumulative_returns_TC.plot(figsize=[15,10], ax=ax)
MD_cumulative_returns_TC.plot(figsize=[15,10], ax=ax)
EW_cumulative_returns.plot(figsize=[15,10], ax=ax)
HRP2_cumulative_returns_TC.plot(figsize=[15,10], ax=ax)

mpl.legend(['HRP', 'Naive RP','RP', 'GMV', 'MD','EW', 'HRP2'])
mpl.ylabel("Accumulated returns (Index 1)")
mpl.yscale("log")
mpl.savefig('Cumulative returns TC')
mpl.show()
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Annualized turnover
HRP_Turnover_ann = HRP_Turnover.groupby(HRP_Turnover.index // 12).sum()
HRP2_Turnover_ann = HRP2_Turnover.groupby(HRP2_Turnover.index // 12).sum()
IVP_Turnover_ann = IVP_Turnover.groupby(IVP_Turnover.index // 12).sum()
GMV_Turnover_ann = GMV_Turnover.groupby(GMV_Turnover.index // 12).sum()
MD_Turnover_ann = MD_Turnover.groupby(MD_Turnover.index // 12).sum()
RP_Turnover_ann = RP_Turnover.groupby(RP_Turnover.index // 12).sum()
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Turnover table
Turnover_table = [[HRP_Turnover_ann.max(), HRP2_Turnover_ann.max(), IVP_Turnover_ann.max(),
                   GMV_Turnover_ann.max(), MD_Turnover_ann.max(), RP_Turnover_ann.max()]]
Turnover_table.append([HRP_Turnover_ann.min(), HRP2_Turnover_ann.min(), IVP_Turnover_ann.min(),
                       GMV_Turnover_ann.min(), MD_Turnover_ann.min(), RP_Turnover_ann.min()])
Turnover_table.append([HRP_Turnover_ann.mean(), HRP2_Turnover_ann.mean(), IVP_Turnover_ann.mean(),
                       GMV_Turnover_ann.mean(), MD_Turnover_ann.mean(), RP_Turnover_ann.mean()])
Turnover_table.append([HRP_TC.sum(), HRP2_TC.sum(), IVP_TC.sum(), GMV_TC.sum(), MD_TC.sum(), RP_TC.sum()])
Turnover_table = pd.DataFrame(Turnover_table, columns=['HRP', 'HRP2','Naive RP','GMV','MD','Risk Parity'],
                              index=['Max. turnover', 'Min. turnover', 'Avg. turnover',
                                     'total transaction cost'])
Turnover_table
\end{lstlisting}

\begin{lstlisting}[language=Python]
############# Monte Carlo experiment
import time

def generateData(nObs,sLength,size0,size1,mu0,sigma0,sigma1F):
    # Time series of correlated variables
    #1) generate random uncorrelated data
    x=np.random.normal(mu0,sigma0,size=(nObs,size0)) # each row is a variable 
    #2) create correlation between the variables
    cols=[random.randint(0,size0-1) for i in range(size1)]
    y=x[:,cols]+np.random.normal(0,sigma0*sigma1F,size=(nObs,len(cols)))
    x=np.append(x,y,axis=1)
    #3) add common random shock
    point=np.random.randint(sLength,nObs-1,size=2)
    x[np.ix_(point,[cols[0],size0])]=np.array([[-.5,-.5],[2,2]])
    #4) add specific random shock
    point=np.random.randint(sLength,nObs-1,size=2)
    x[point,cols[-1]]=np.array([-.5,2])
    return x,cols

def getHRP(cov,corr):
    # Construct a hierarchical portfolio
    corr,cov=pd.DataFrame(corr),pd.DataFrame(cov)
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link) 
    sortIx=corr.index[sortIx].tolist() # recover labels 
    hrp=getRecBipart(cov,sortIx)
    return hrp.sort_index()

def getHRP2(cov,corr):
    # Construct a hierarchical portfolio
    corr,cov=pd.DataFrame(corr),pd.DataFrame(cov)
    dist=correlDist(corr) 
    link=sch.linkage(dist,'single')
    cluster_dict = get_cluster_dict(link)
    hrp2 = recClusterVar(cluster_dict,link, cov)
    return hrp2


def hrpMC(numIters=20,nObs=520,size0=5,size1=5,mu0=0,sigma0=0.01, sigma1F=.25,sLength=260,rebal=22):
    start_time = time.time()
    # Monte Carlo experiment on HRP
    methods=[getHRP,getHRP2,getIVP,risk_parity,
             GMVPortfolio,GMVLOPortfolio,max_div_port,ewPortfolio]#,getCLA] 
    stats,numIter={i.__name__:pd.Series() for i in methods},0
    pointers=range(sLength,nObs,rebal)
    divratio={i.__name__:pd.Series() for i in methods}
    rc={i.__name__:pd.Series() for i in methods}
    w={i.__name__:pd.DataFrame() for i in methods}
    while numIter<numIters:
        #1) Prepare data for one experiment 
        x,cols=generateData(nObs,sLength,size0,size1,mu0,sigma0,sigma1F)
        r={i.__name__:pd.Series() for i in methods}
        #2) Compute portfolios in-sample
        for pointer in pointers:
            x_=x[pointer-sLength:pointer]
            cov_,corr_=np.cov(x_,rowvar=0),np.corrcoef(x_,rowvar=0) 
            #3) Compute performance out-of-sample
            x_=x[pointer:pointer+rebal]
            cov_out=pd.DataFrame(np.cov(x[pointer:pointer+rebal],rowvar=0))
            for func in methods:
                w_=pd.Series(func(cov=cov_,corr=corr_))
                # callback
                w[func.__name__]=w[func.__name__].append(w_,ignore_index=True)
                r_=pd.Series(np.dot(x_,w_))
                divratio_ = calc_diversification_ratio(w_,cov_out)
                divratio[func.__name__]=divratio[func.__name__].append(pd.Series(divratio_))
                rc_ = np.squeeze(np.asarray(calculate_risk_contribution(w_, cov_out)))
                        /np.sqrt(calculate_portfolio_var(w_,cov_out))
                rc[func.__name__]=rc[func.__name__].append(pd.Series(rc_))
                r[func.__name__]=r[func.__name__].append(r_)
        #4) Evaluate and store results
        for func in methods:
            r_=r[func.__name__].reset_index(drop=True)
            p_=(1+r_).cumprod()
            stats[func.__name__].loc[numIter]=p_.iloc[-1]-1 # terminal return
        numIter+=1
    #5) Report results
    stats=pd.DataFrame.from_dict(stats,orient='columns')
    stats.to_csv('stats.csv')
    df0,df1=stats.std(),stats.var()
    print(pd.concat([df0,df1,df1/df1['getHRP']-1],axis=1))
    return divratio,rc, w, r
        
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Results of Monte Carlo experiment
divratios, rcs, w, r = hrpMC()
\end{lstlisting}

\begin{lstlisting}[language=Python]
#Function for calculating MDRC
def portfolio_risk_deviance(rcs,mean=True):
    deviances=[]
    for i in rcs:
        sam = 0
        for j in i:
            sam = sam+abs(j-1/len(i))
        deviances.append(sam)
    if mean:    
        return np.array(deviances).mean()
    else:
        return np.array(deviances)

\end{lstlisting}

