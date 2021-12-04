import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import io, base64, os, json, re, glob
import datetime
from datetime import timedelta
import pandas as pd
import pydata_google_auth
import numpy as np

from fbprophet import Prophet
import statsmodels.api as sm


df_lamb = pd.read_csv('/Users/ermiyasliyeh/Downloads/lamb.csv')
df_lamb['ds'] = pd.to_datetime(df_lamb['ds'])
df_lamb = df_lamb.sort_values('ds', ascending=True)
print(df_lamb.shape)
print(df_lamb.head())
print(df_lamb.tail())

train_dataset = df_lamb.copy()
prophet_basic = Prophet()
prophet_basic.fit(train_dataset)
future = prophet_basic.make_future_dataframe(periods=365)
future.tail()

forecast = prophet_basic.predict(future)
# plotting the predicted data
fig1 = prophet_basic.plot(forecast)
fig1.savefig('lamb_output1')

fig2 = prophet_basic.plot_components(forecast)
fig2.savefig('lamb_output2')
# ---------------------------------------------

df_beef = pd.read_csv('/Users/ermiyasliyeh/Downloads/beef.csv')
df_beef['ds'] = pd.to_datetime(df_beef['ds'])
df_lamb = df_beef.sort_values('ds', ascending=True)
print(df_beef.shape)
print(df_beef.head())
print(df_beef.tail())

train_dataset = df_beef.copy()
prophet_basic = Prophet()
prophet_basic.fit(train_dataset)
future = prophet_basic.make_future_dataframe(periods=365)
future.tail()

forecast = prophet_basic.predict(future)
# plotting the predicted data
fig1 = prophet_basic.plot(forecast)
fig1.savefig('beef_output1')

fig2 = prophet_basic.plot_components(forecast)
fig2.savefig('beef_output2')
# ---------------------------------------------

df_cheese = pd.read_csv('/Users/ermiyasliyeh/Downloads/cheese.csv')
df_cheese['ds'] = pd.to_datetime(df_cheese['ds'])
df_lamb = df_cheese.sort_values('ds', ascending=True)
print(df_cheese.shape)
print(df_cheese.head())
print(df_cheese.tail())

train_dataset = df_cheese.copy()
prophet_basic = Prophet()
prophet_basic.fit(train_dataset)
future = prophet_basic.make_future_dataframe(periods=365)
future.tail()

forecast = prophet_basic.predict(future)
# plotting the predicted data
fig1 = prophet_basic.plot(forecast)
fig1.savefig('cheese_output1')

fig2 = prophet_basic.plot_components(forecast)
fig2.savefig('cheese_output2')
# ---------------------------------------------

df_pork = pd.read_csv('/Users/ermiyasliyeh/Downloads/pork.csv')
df_pork['ds'] = pd.to_datetime(df_pork['ds'])
df_lamb = df_pork.sort_values('ds', ascending=True)
print(df_pork.shape)
print(df_pork.head())
print(df_pork.tail())

train_dataset = df_pork.copy()
prophet_basic = Prophet()
prophet_basic.fit(train_dataset)
future = prophet_basic.make_future_dataframe(periods=365)
future.tail()

forecast = prophet_basic.predict(future)
# plotting the predicted data
fig1 = prophet_basic.plot(forecast)
fig1.savefig('pork_output1')

fig2 = prophet_basic.plot_components(forecast)
fig2.savefig('pork_output2')
# ---------------------------------------------

df_farmedSalmon = pd.read_csv('/Users/ermiyasliyeh/Downloads/farmedSalmon.csv')
df_farmedSalmon['ds'] = pd.to_datetime(df_farmedSalmon['ds'])
df_lamb = df_farmedSalmon.sort_values('ds', ascending=True)
print(df_farmedSalmon.shape)
print(df_farmedSalmon.head())
print(df_farmedSalmon.tail())

train_dataset = df_farmedSalmon.copy()
prophet_basic = Prophet()
prophet_basic.fit(train_dataset)
future = prophet_basic.make_future_dataframe(periods=365)
future.tail()

forecast = prophet_basic.predict(future)
# plotting the predicted data
fig1 = prophet_basic.plot(forecast)
fig1.savefig('farmedSalmon_output1')

fig2 = prophet_basic.plot_components(forecast)
fig2.savefig('farmedSalmon_output2')
# ---------------------------------------------

df_turkey = pd.read_csv('/Users/ermiyasliyeh/Downloads/turkey.csv')
df_turkey['ds'] = pd.to_datetime(df_turkey['ds'])
df_lamb = df_turkey.sort_values('ds', ascending=True)
print(df_turkey.shape)
print(df_turkey.head())
print(df_turkey.tail())

train_dataset = df_turkey.copy()
prophet_basic = Prophet()
prophet_basic.fit(train_dataset)
future = prophet_basic.make_future_dataframe(periods=365)
future.tail()

forecast = prophet_basic.predict(future)
# plotting the predicted data
fig1 = prophet_basic.plot(forecast)
fig1.savefig('turkey_output1')

fig2 = prophet_basic.plot_components(forecast)
fig2.savefig('turkey_output2')
# ---------------------------------------------

