import numpy as np
import pickle
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


sorular = [
    "bu ay işlem kaç sayı sayısı adet adeti",
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


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sorular)
y = cevaplar

# Alternatif, şu an kullanılmıyoe
clf = MultinomialNB()
clf.fit(X, y)


# Örnek bir tahmin yapalım          
ornek_soru = ["1b"]
ornek_soru_v = vectorizer.transform(ornek_soru) 

tahmin = clf.predict(ornek_soru_v)[0]

proba = clf.predict_log_proba(ornek_soru_v)
proba
print("Örnek Tahmin:", tahmin)

cousine_scores = []
for i in range(len(ds)):
    cousine_score =  cosine_similarity(ornek_soru_v, vectorizer.transform(ds[['Soru']].iloc[i,:]))
    cousine_scores.append(cousine_score)

score_index = cousine_scores.index(max(cousine_scores))
cevaplar[score_index]
                      
cevaplar[np.argmax(cousine_scores)]
np.max(cousine_scores)
cousine_scores
proba


pickle.dump(clf, open('model.pkl','wb'))
pickle.dump(vectorizer, open('vectorizer.pkl','wb'))
