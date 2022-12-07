import pandas as pd
import numpy as np
from functools import reduce

# csak azért hogy meglegyen az eredeti táblázat
df_dba_original = pd.read_csv('DBA.csv')
df_tlt_original = pd.read_csv('TLT.csv')
df_vde_original = pd.read_csv('VDE.csv')
df_xlv_original = pd.read_csv('XLV.csv')
df_xme_original = pd.read_csv('XME.csv')

df_dba = df_dba_original[['Date', 'Adj Close']]
df_tlt = df_tlt_original[['Date', 'Adj Close']]
df_vde = df_vde_original[['Date', 'Adj Close']]
df_xlv = df_xlv_original[['Date', 'Adj Close']]
df_xme = df_xme_original[['Date', 'Adj Close']]

df_dba = df_dba.rename(columns={'Adj Close': 'Adj Close_dba'})
df_tlt = df_tlt.rename(columns={'Adj Close': 'Adj Close_tlt'})
df_vde = df_vde.rename(columns={'Adj Close': 'Adj Close_vde'})
df_xlv = df_xlv.rename(columns={'Adj Close': 'Adj Close_xlv'})
df_xme = df_xme.rename(columns={'Adj Close': 'Adj Close_xme'})

filenames = [df_dba, df_tlt, df_vde, df_xlv, df_xme]


df_merge = reduce(lambda left, right: pd.merge(left, right, on=['Date'], how='inner'), filenames)


pass
