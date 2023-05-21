import pandas as pd
import numpy as np
import scipy as sp
from functools import reduce
import matplotlib.pyplot as plt

# fájl beolvasás és tisztítás
df_dba_original = pd.read_csv('DBA.csv', index_col=0)
df_tlt_original = pd.read_csv('TLT.csv', index_col=0)
df_vde_original = pd.read_csv('VDE.csv', index_col=0)
df_xlv_original = pd.read_csv('XLV.csv', index_col=0)
df_xme_original = pd.read_csv('XME.csv', index_col=0)
df_risk_free_rate = pd.read_csv('../Projekt_Numpu/DTB3.csv')
df_risk_free_rate.index = pd.to_datetime(df_risk_free_rate['DATE'])
df_risk_free_rate = df_risk_free_rate[['DTB3']]
df_risk_free_rate.columns = ['risk_free']
msk = df_risk_free_rate['risk_free'] != '.'
df_risk_free_rate = df_risk_free_rate[msk]
df_risk_free_rate = df_risk_free_rate.astype(float)


df_dba = df_dba_original[['Adj Close']]
df_tlt = df_tlt_original[['Adj Close']]
df_vde = df_vde_original[['Adj Close']]
df_xlv = df_xlv_original[['Adj Close']]
df_xme = df_xme_original[['Adj Close']]

df_dba = df_dba.rename(columns={'Adj Close': 'Adj Close_dba'})
df_tlt = df_tlt.rename(columns={'Adj Close': 'Adj Close_tlt'})
df_vde = df_vde.rename(columns={'Adj Close': 'Adj Close_vde'})
df_xlv = df_xlv.rename(columns={'Adj Close': 'Adj Close_xlv'})
df_xme = df_xme.rename(columns={'Adj Close': 'Adj Close_xme'})

filenames = [df_dba, df_tlt, df_vde, df_xlv, df_xme]

df_merge = reduce(lambda left, right: pd.merge(left, right, on=['Date'], how='inner'), filenames)
df_merge.index = pd.to_datetime(df_merge.index)
df_minden = df_merge.join(df_risk_free_rate, how='inner')
df_minden.iloc[:, -1] = df_minden.iloc[:, -1] /100

def calc_nasset_mean(w, mean_return):
    return np.sum(w*mean_return)


def calc_nasset_std(w, cov_matrix):
    return np.sqrt(np.dot(np.dot(w, cov_matrix), w.transpose()))


days_in_year = 252
years_roll = 5
start = years_roll*days_in_year

# asset metrics
risk_free_rate = df_minden.iloc[:, -1].mean()
return_asset = df_merge / df_merge.shift(1) - 1
mean_asset = return_asset.mean() * 12
std_asset = return_asset.std() * np.sqrt(12)
cov_asset = return_asset.cov() * 12
corr_asset = return_asset.corr()

# Sharpe-mutató = (Portfólió hozama – Kockázatmentes hozam)/ Portfólió szórása


def negSharpe(w, riskfreerate, cov_matrix, mean_return):
    return -1 * ((calc_nasset_mean(w, mean_return)-riskfreerate)/calc_nasset_std(w, cov_matrix))


cons = ({'type': 'eq', 'fun': lambda weight: np.sum(weight)-1})
bounds = []

for i in range(mean_asset.shape[0]):
    bounds.append((-10, 10))

SHres = sp.optimize.minimize(negSharpe, np.array([0, 0, 0, 1, 0]), args=(risk_free_rate, cov_asset, mean_asset),
                             constraints=cons, bounds=bounds)

eredmenySH = SHres.x
SharpeMax = -1*negSharpe(eredmenySH, risk_free_rate, cov_asset, mean_asset)

def Sharpe_roll(w, df_minden, time, dist):
    df_dist = return_asset.iloc[time:time+dist-1]
    a = df_dist.mean()
    szoras =a.std()
    kock=df_minden.iloc[:, -1].mean()
    return -1*((calc_nasset_mean(w, a)-kock)-np.sum(w*szoras))


SH_roll = []
for napok in range(len(df_merge)-2*start):
    SH_roll.append(sp.optimize.minimize(Sharpe_roll, np.array([0, 0, 0, 1, 0]), args=(df_minden, napok, start),
                             constraints=cons, bounds=bounds).x)
SH_roll = pd.DataFrame(SH_roll)
SH_roll.plot()
plt.show()

# új feladatrész
bounds_MDD = []
for i in range(mean_asset.shape[0]):
    bounds_MDD.append((0, 1))


def calc_nasset_MDD(w, df, window):
    Roll_Max = df.rolling(window, min_periods=1).max()
    Daily_Drawdown = df / Roll_Max - 1.0
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
    MDD = Max_Daily_Drawdown.iloc[-1]
    return -1*np.sum(w*MDD)


MDDres = sp.optimize.minimize(calc_nasset_MDD, np.array([1, 0, 0, 0, 0]), args=(df_merge, len(df_merge.index)),
                              constraints=cons, bounds=bounds_MDD)
eredmenymdd = MDDres.x
minMDD = -1 * calc_nasset_MDD(eredmenymdd, df_merge, len(df_merge.index))


def t_drawdown_min(w, df, time, dist):
    df_dist = df.iloc[time:time+dist-1]
    Roll_Max_tmin = df_dist.rolling(len(df_dist), min_periods=1).max()
    Daily_Drawdown_tmin = df_dist / Roll_Max_tmin - 1.0
    Max_Daily_Drawdown_tmin = Daily_Drawdown_tmin.rolling(len(df_dist), min_periods=1).min()
    MDD_tmin = Max_Daily_Drawdown_tmin.iloc[-1]
    return -1 * np.sum(w * MDD_tmin)


MDDrollres = []
for days in range(len(df_merge)-(2*start)):
    MDDrollres.append(sp.optimize.minimize(t_drawdown_min, np.array([1, 0, 0, 0, 0]), args=(df_merge, days, start),
                                      constraints=cons, bounds=bounds_MDD).x)

MDD_roll = pd.DataFrame(MDDrollres)
MDD_roll.plot()
plt.show()
 print(1)
 print(2)
pass

