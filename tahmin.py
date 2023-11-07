#!pip install yfinance
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostRegressor



df = pd.read_csv('C:/Users/ASUS/Documents/ödeal chatbot/hackathon_data.csv')
df = df[df.id == 301002470]
df['tarih'] = pd.to_datetime(df['IslemTarih'])
df = df.set_index(df.tarih)
df['year'] = df.index.year
df['month'] = df.index.month
df['day_of_w'] = df.index.dayofweek
df['day_of_y'] = df.index.day_of_year
df['week'] = df.index.isocalendar().week
df['hour'] = df.index.hour
df['day'] = df.index.day



train = df[["year","month","day_of_w","day_of_y","week","day","IslemTutar"]]

train = train.groupby(by=["year","month","day"])['IslemTutar'].sum().reset_index()

train['date'] = pd.to_datetime(train[['year', 'month', 'day']])
train = train.set_index(train.date)
train['day_of_w'] = train.index.dayofweek
train['day_of_y'] = train.index.day_of_year
train['week'] = train.index.isocalendar().week

train['isweekend']=  train.day_of_w.apply(lambda x : 0 if x < 5 else 1)




for i in [1,2,3,4,5,6,7,10,14,30]:
    column_name = f'lag_mean{i}'
    train[column_name] = train['IslemTutar'].shift(14).rolling(i).mean()

togettime = train.sort_values(by=['year',"month","day"]).head(1)

import datetime
dolar_tl = yf.Ticker("USDTRY=X")

start_date = datetime.datetime(togettime.year.values[0], togettime.month.values[0], togettime.day.values[0])

end_date = datetime.datetime.now()

kur_data = dolar_tl.history(period="1d", start=start_date, end=end_date)["Close"]
kurdf  = pd.DataFrame(kur_data)


kurdf["year"]  =  kurdf.index.year
kurdf["month"]  =  kurdf.index.month
kurdf["day"]  =  kurdf.index.day


train = train.merge(kurdf, on=["year","month","day"],how="left")
train['dolar'] = train.Close.shift(14)

train["dolar"] =  train.dolar.fillna(method="ffill")

for i in [1,2,3,4,5,6,7,10,14,30]:
    column_name = f'dolar_mean{i}'
    train[column_name] = train['dolar'].rolling(i).mean()
train.drop("Close", axis=1, inplace=True)


train["target"] = np.log(train.IslemTutar)




from sklearn.model_selection import TimeSeriesSplit

fold = 0
scores = []
preds =[]


tscv = TimeSeriesSplit(n_splits=3,test_size=100)  
for train_idx, val_idx in tscv.split(train):
    train_data = train.iloc[train_idx]
    test_data = train.iloc[val_idx]

    X_train, y_train = train_data.drop('target', axis=1), train_data['target']
    X_test, y_test = test_data.drop('target', axis=1), test_data['target']
    
    
    features = ['year', 'month', 'day',  'day_of_w', 'day_of_y',
       'week', 'lag_mean1', 'lag_mean2', 'lag_mean3',
       'lag_mean4', 'lag_mean5', 'lag_mean6', 'lag_mean7', 'lag_mean10',
       'lag_mean14', 'lag_mean30', 'dolar',  
       ]       
    X_train = X_train[features]
    X_test =  X_test[features]
    
    
    
    
    reg = CatBoostRegressor(silent = True)
    print(f'{fold + 1}. Fold Training... ')
    fold += 1
    
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    preds.append(pred)
    
    score = mean_absolute_error(np.exp(y_test),np.exp(pred))
    scores.append(score)
    
    print(score)
mean_score = sum(scores) / len(scores)
print("Mean Score:", mean_score)





# Bugünkü tarihi alın
bugun = datetime.date.today()

# 14 gün sonrasını hesaplayın
son_tarih = bugun + datetime.timedelta(days=14)


tarihler = pd.date_range(start=bugun, end=son_tarih, freq='D')

# Bu tarihleri içeren bir DataFrame oluşturun
tarih_df = pd.DataFrame({'Tarih': tarihler})
tdf = tarih_df.set_index(tarih_df.Tarih)



tdf['year'] = tdf.index.year
tdf['month'] = tdf.index.month
tdf['day_of_w'] = tdf.index.dayofweek
tdf['day_of_y'] = tdf.index.day_of_year
tdf['week'] = tdf.index.isocalendar().week
tdf['day'] = tdf.index.day
tdf['isweekend']=  tdf.day_of_w.apply(lambda x : 0 if x < 5 else 1)


import datetime
dolar_tl = yf.Ticker("USDTRY=X")

start_date = datetime.datetime(togettime.year.values[0], togettime.month.values[0], togettime.day.values[0])

end_date = datetime.datetime.now()

kur_data = dolar_tl.history(period="1d", start=start_date, end=end_date)["Close"]
kurdf  = pd.DataFrame(kur_data)


kurdf["year"]  =  kurdf.index.year
kurdf["month"]  =  kurdf.index.month
kurdf["day"]  =  kurdf.index.day


dolar = train.merge(kurdf, on=["year","month","day"],how="left")
tdf['dolar'] =  dolar.dolar[-15:].reset_index(drop=True).values



t_df = df[["year","month","day_of_w","day_of_y","week","day","IslemTutar"]]
t_df = t_df.groupby(by=["year","month","day"])['IslemTutar'].sum().reset_index()


for i in [1,2,3,4,5,6,7,10,14,30]:
    column_name = f'lag_mean{i}'
    t_df[column_name] = t_df['IslemTutar'].rolling(i).mean()
    


tomergetdf = t_df.sort_values(by=['year', "month", "day"])[['lag_mean1', 'lag_mean2', 'lag_mean3',
       'lag_mean4', 'lag_mean5', 'lag_mean6', 'lag_mean7', 'lag_mean10',
       'lag_mean14', 'lag_mean30']].iloc[-15:]

newdays = pd.concat([tdf.reset_index(drop=True),tomergetdf.reset_index(drop=True)],axis=1)
cb = CatBoostRegressor(silent=True)
features =  ['year', 'month', 'day',  'day_of_w', 'day_of_y',
   'week', 'lag_mean1', 'lag_mean2', 'lag_mean3',
   'lag_mean4', 'lag_mean5', 'lag_mean6', 'lag_mean7', 'lag_mean10',
   'lag_mean14', 'lag_mean30', 'dolar',  
   ] 
cb = cb.fit(train[features], train.target)
cb_preds = np.exp(cb.predict(X_test))



