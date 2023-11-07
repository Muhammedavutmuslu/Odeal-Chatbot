import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session,url_for
import pickle
import time 
import json
import matplotlib.pyplot as plt
import flask
import io
import os
import base64
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import yfinance
# df düzenleme

df = pd.read_csv("C:/Users/Muhammed/Documents/ödeal/hackathon_data.csv")
df = df[df.id == 301002470]
df['tarih'] = pd.to_datetime(df['IslemTarih'])
df = df.set_index(df.tarih)
df['year'] = df.index.year
df['month'] = df.index.month
df['day_of_w'] = df.index.dayofweek
df['day_of_y'] = df.index.day_of_year
df['week'] = df.index.isocalendar().week
df['hour'] = df.index.hour

df.loc[(df.year==2023) & (df.week ==52) & (df.month=='Ocak'), 'week'] = 1
df.drop("IslemTarih",axis=1,inplace=True)

aylar = {
    1: 'Ocak',
    2: 'Şubat',
    3: 'Mart',
    4: 'Nisan',
    5: 'Mayıs',
    6: 'Haziran',
    7: 'Temmuz',
    8: 'Ağustos',
    9: 'Eylül',
    10: 'Ekim',
    11: 'Kasım',
    12: 'Aralık'
}

df["month"] =  df['month'].map(aylar)


örn = df.iloc[:1000,:]

#---------------------------------------------------

#                       fonksiyonlar 




def sonayişlemadet():
    sonay = df.reset_index(drop=True).sort_values(by="tarih").tail(1)['month'].values[0]
    adet = df[df.month == sonay]["IslemTutar"].count().round()
    return adet

def sonyılişlemadet():
    year = df.reset_index(drop=True).sort_values(by="tarih").tail(1)['year'].values[0]
    adet = df[df.year == year]["IslemTutar"].count().round()
    return adet




def sonaytoplam():
    sonay = df.reset_index(drop=True).sort_values(by="tarih").tail(1)['month'].values[0]
    tutar = df[df.month == sonay]["IslemTutar"].sum().round()
    return tutar
    

def aylıktoplam():
    tutar = df[["month","IslemTutar"]].groupby(by='month').sum().round()
    return tutar

def tümtoplam():
    tutar = df.IslemTutar.sum().round()
    return tutar
    
def ilktoplam():
    tutar  = df.head().IslemTutar.sum().round()
    return tutar 

def haftalıkoran():   ####
    year = df.reset_index(drop=True).sort_values(by="tarih").tail(1)['year'].values[0]
    fdf = df[df.year==year]
    tutar = ((1 - (fdf.groupby(by=["year",'week'])['IslemTutar'].sum()[:-1].reset_index()  /  fdf.groupby(by=["year",'week'])['IslemTutar'].sum()[1:].reset_index()))*100)['IslemTutar']
    tutar = tutar.to_string(index=True, header=False)
    return tutar

def aktivasyontarih():
    akt = df.UyeAktivasyonTarih.sort_values().tail(1).values[0]
    return akt



def yıllıkgrafik():
    plt.figure()
    means = df.groupby(by='year')['IslemTutar'].mean().reset_index()
    sns.barplot(means,x='year',y='IslemTutar', color='blue')
    plt.title("Yıllara göre satış toplamları")
    plt.xlabel("Yıl")
    plt.ylabel("Tutar")
    plt.savefig('static/Image/yillik_grafik.png')


def haftalık_degisim():
    plt.figure()
    year = df.reset_index(drop=True).sort_values(by="tarih").tail(1)['year'].values[0]
    fdf = df[df.year==year]
    tutar = ((1 - (fdf.groupby(by=["year",'week'])['IslemTutar'].mean()[:-1].reset_index()  /  fdf.groupby(by=["year",'week'])['IslemTutar'].mean()[1:].reset_index()))*100)['IslemTutar']
    tutar = tutar.round()
    sns.barplot(y=tutar[:-1] , x = tutar.index[:-1], color='blue')
    plt.title("Son yıl içerisinde haftalık İşlem tutar değişimi")
    plt.xlabel("Hafta")
    plt.ylabel("değişim %")
    plt.savefig('static/Image/yillik_grafik.png')
 

def aylıkgrafik():
    plt.figure()
    means = df.groupby(by='month')['IslemTutar'].sum().reset_index()
    sns.barplot(means,y='month',x='IslemTutar', color='blue' ,order=["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"])
    plt.title("Aylara göre satış toplamları")
    plt.ylabel("İşlem Tutarı")
    plt.xlabel("İşlem Tarihi")
    plt.savefig('static/Image/yillik_grafik.png')


def tercihgrafik():
    tc = df.Tercih.value_counts().reset_index()
    plt.pie(tc['count'], labels=tc['Tercih'], autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  
    plt.title('Tercih Dağılımı')
    plt.savefig('static/Image/yillik_grafik.png')
    
def saatlikgrafik():
    hourds = df.groupby(by='hour')['IslemTutar'].sum().reset_index()
    sns.lineplot(hourds, x = "hour", y = "IslemTutar")
    plt.title("Saatlik İşlem Tutarı")
    plt.xlabel("saat")
    plt.ylabel("işlem tutarı")
    plt.savefig('static/Image/yillik_grafik.png')


#---------------------------------------------------------

sorular = [
    "bu ay işlem kaç sayı adet adeti  sayısı",
    "bu yıl işlem kaç sayı sayısı",
    "tümünü tüm hepsi topla",
    "ilk",
    "aylık   ay 12 ",
    "bu ay en bu aydaki  geçen toplamı toplam",
    "aktivasyon aktivasyonum aktiv tarih tarihi",
    "yıllık satış grafik grafiği ",
    "haftalık haftalara satış işlem değişim haftadan grafiği grafik grafiğini görsel görselini",
    "aylık  aylara grafik grafiği grafik grafiğini görsel görselini",
    "tercih tercihi işlem türü şekli",
    "saatlik saatlere saat bazında saatler",
    "tahmini tahmin model modeli tahminleri"
   
]


cevaplar = [
            "sonayişlemadet",
            "sonyılişlemadet",
            "tümtoplam",
            "ilktoplam",
            "aylıktoplam",
            "sonaytoplam",
            "aktivasyontarih",
            "yıllıkgrafik",
            "haftalık_degisim",
            "aylıkgrafik",
            "tercihgrafik",
            "saatlikgrafik",
            "tahminmodeli"
            
            
]




ds = pd.DataFrame({'Soru': sorular, 'Cevap': cevaplar})


örn = df.iloc[:1000,:]

app = Flask(__name__)

app.secret_key = '43324'

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    session['messages'] = []
    add_message("bot", """first""")
    
    
    
    return render_template('index.html', messages=session['messages'])



@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    user_input = user_input.lower()

   
    user_input_array = vectorizer.transform([user_input])

    
    cousine_scores = []
    for i in range(len(ds)):
        cousine_score = cosine_similarity(user_input_array, vectorizer.transform([ds['Soru'].iloc[i]]))
        cousine_scores.append(cousine_score)
  
    score_index = cousine_scores.index(max(cousine_scores))
    prediction =  cevaplar[score_index]
    
    grafik = []
    
    if max(cousine_scores) == 0:
        total = "üzgünüm anlamadım sormak istediğiniz soruyu biraz daha farklı bir şekilde sorar mısınız?"
        add_ununswered(user_input)
    
        
    elif prediction == "sonayişlemadet":
        total =  sonayişlemadet()
        total = f"Bu ayda gerçekleşen işlem adet sayısı: {total}"
        
    elif prediction == "sonayılişlemadet":
        total =  sonyılişlemadet()
        total = f"Bu yıl erçekleşen işlem adet sayısı: {total}"
    
    
    elif prediction == "tümtoplam":
        total =  tümtoplam()
        total = f"tüm satışlarınızın toplamı: {total}"
    
    elif prediction == "aylıktoplam":
        total = aylıktoplam()
        total = total.to_string(index=True, header=False)
        
        
    elif prediction == "sonaytoplam":
        total = sonaytoplam()
        total = f"son aydaki satışlarınızın toplamı: {total}"
        
        
    elif prediction == "aktivasyontarih":
         total = aktivasyontarih()
         total = f"son aktivasyon tarihiniz: {total}"
    
    
    elif prediction == "tercihgrafik":
        total = tercihgrafik()
        total = "Solda ödeme türlerini görebilirsin"
        
        
    elif prediction == "yıllıkgrafik":
         grafik = yıllıkgrafik()
         total = """sol tarafta grafiği görebilirsiniz.DİKKAT:Son yılın sonuna 
                       gelmediğimiz için diğer yıllara göre az olabilir"""
      
        
    elif prediction == "haftalık_degisim":
          grafik = haftalık_degisim()
          total = """sol tarafta grafiği görebilirsiniz."""
          
    
    elif prediction == "aylıkgrafik":
          grafik = aylıkgrafik()
          
          total = """sol tarafta grafiği görebilirsiniz.DİKKAT: son ay henüz tamamlanmadığı için 
                     diğer aylara göre işlem tutarı farklı olabilir"""
                     
    elif prediction == "saatlikgrafik":
          grafik = saatlikgrafik()
          total = """sol tarafta grafiği görebilirsiniz.En çok işlemin gerçekleştiği saatleri görebilrisiniz."""
          
          
    elif prediction == "tahminmodeli":
           tahmin = tahminmodeli()
           total = """Sıfır bugün olmak üzere 14 günlük tahminler grafiksel olarak verilmiştir. tahminler sadece fikir vermek amacıyla oluşturulmuştur.Gerçek değerler tahminden oldukça farklı olabilir."""    
           sns.barplot(y=tahmin.preds,x=tahmin.index)
           plt.title("14 günlük tahminler")
           plt.xlabel("günler")
           plt.ylabel("tahmini toplam işlem tutarı")
           plt.savefig('static/Image/yillik_grafik.png')
        
    else:
        total = ilktoplam()
        total = f"ilk 5 günün satış miktarı: {total}"
        
    

    
    add_message("user", user_input)
    add_message("bot", total)

    return render_template("index.html",messages=session['messages'], chart_data = grafik)

def add_message(type, content):
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M")  # Saat ve dakika formatı
    message = {'type': type, 'content': content, 'timestamp': timestamp}
    messages = session.get('messages', [])
    messages.append(message)
    session['messages'] = messages
   

def add_ununswered(user_input):
    with open('ununswered_inputs.txt',  'a',  encoding="utf-8") as f:
        f.write(f'{user_input}\n')
        

        
def tahminmodeli():
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import KFold, cross_val_score
    from catboost import CatBoostRegressor



    df = pd.read_csv('C:/Users/Muhammed/Documents/ödeal/hackathon_data.csv')
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



    
    bugun = datetime.date.today()

    
    son_tarih = bugun + datetime.timedelta(days=14)


    tarihler = pd.date_range(start=bugun, end=son_tarih, freq='D')

    
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
    cb_preds = np.exp(cb.predict(newdays))
    newdays["tarih"] = pd.to_datetime(newdays[["year", "month","day" ]])
    newdays["preds"] = cb_preds
    last = newdays[["tarih","preds"]]
    return last
    



if __name__ == "__main__":
    app.run(debug=True)

