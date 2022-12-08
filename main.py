import pandas as pd
import numpy as np
import scipy as sp
from functools import reduce


# csak azért hogy meglegyen az eredeti táblázat
df_dba_original = pd.read_csv('DBA.csv', index_col=0)
df_tlt_original = pd.read_csv('TLT.csv', index_col=0)
df_vde_original = pd.read_csv('VDE.csv', index_col=0)
df_xlv_original = pd.read_csv('XLV.csv', index_col=0)
df_xme_original = pd.read_csv('XME.csv', index_col=0)

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

def calc_nasset_mean(w, mean_return):
    return np.sum(w*mean_return)

def calc_nasset_std(w, cov_matrix):
    return np.sqrt(np.dot(np.dot(w, cov_matrix), w.transpose()))


def calc_nasset_mean_std(w,mean_return,cov_matrix):
    ret = calc_nasset_mean(w, mean_return)
    std = calc_nasset_std(w, cov_matrix)
    return ret, std


w1s = np.linspace(-1, 1, 11)


return_asset = df_merge / df_merge.shift(1) - 1
mean_asset = return_asset.mean() * 12
std_asset = return_asset.std() * np.sqrt(12)
cov_asset = return_asset.cov() * 12
corr_asset = return_asset.corr()

calc_nasset_mean_std(np.array([1, 0, 0, 0, 0]), mean_asset, cov_asset)

grid = np.array(np.meshgrid(w1s, w1s, w1s, w1s))
grid = grid.reshape((4, -1)).transpose()
grid = np.c_[grid, 1-grid.sum(axis=1)]

nsasset_mean_std = []
for i in range(grid.shape[0]):
    ret, std = calc_nasset_mean_std(grid[i], mean_asset, cov_asset)
    nsasset_mean_std.append((ret,std))

nsasset_mean_std_df = pd.DataFrame(nsasset_mean_std)
nsasset_mean_std_df.columns = ["Portfolio Return", "Portfolio Std. Dev."]

cons = ({'type': 'eq', 'fun': lambda weight: np.sum(weight)-1})

#még csak szórás minimalizálásra a megfelelő súlyok
#Sharpe-mutató = (Portfólió hozama – Kockázatmentes hozam)/ Portfólió szórása
res = sp.optimize.minimize(calc_nasset_std, np.array([1, 0, 0, 0, 0]), args=cov_asset, constraints=cons)

eredmeny = res.x
pass