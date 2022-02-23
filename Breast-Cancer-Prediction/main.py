#kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#çıkan hataları ortadan kaldırmak için
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

#veri seti yükleme
veri = pd.read_csv("C:\Breast-Cancer-Prediction\data.csv")
veri.head()

#boş sütunu ve gereksiz sütunun kaldırılması
veri.dropna(axis=1, inplace=True)
veri.info()

# M(malignant)(kötü) ve B(benign)(iyi huylu) değerlerini 1,0 olarak değiştirme
veri.diagnosis.replace({'M':1, 'B':0}, inplace=True)
#teşhislerin sayısı
print(veri["diagnosis"].value_counts())

#aykırı verileri belirleme (korelasyon ile)
korelasyon = veri.corr()
plt.figure(figsize=(20,20))
sb.heatmap(veri.corr(), cmap='YlGnBu', annot = True)
plt.title("Korelasyon Haritası", fontweight = "bold", fontsize=16)

#Özellik ve Etiket seçimi
x=veri.iloc[:,2:].values
y=veri.iloc[:,1].values

#özellikleri sayısallaştırma (isim yerine 0-(özelliksayısı -1) arasında sayı verir)
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=lb.fit_transform(y)

#train-test olarak veriyi ayırma
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#StandartScaler ile Z-score standartlaştırma
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#print(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

#yapay sinir ağları başlatılıyor
model=Sequential()

#giriş (ilk gizli) katmanını
model.add(Dense(16,activation='relu', input_dim=30))
#overfitting önlemek için dropout
model.add(Dropout(0.2))
#ikinci gizli katman
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.2))

#çıktı katmanının eklenmesi
model.add(Dense(1, activation='sigmoid')) #sigmoid(x) = 1 / (1 + exp(-x))
model.summary()

#YSA yapılandırması
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#YSA'nın eğitilmesi
history = model.fit(x_train, y_train, batch_size=80, epochs=100, validation_split=0.2)

#test seti sonuçlarını tahmin etme
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)
print(y_pred)

#doğruluk değeri
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)
print('score is:',score*100)

#karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=[14,7])
sb.heatmap(cm,annot=True)
plt.show()

# tüm verilerin listesi
print(history.history.keys())
# doğruluk grafiği
plt.plot(history.history['accuracy'],"--")
plt.plot(history.history['val_accuracy'])
plt.title('Modelin Eğitim Performansı')
plt.ylabel('accuracy')
plt.xlabel('epoch (devir)')
plt.legend(['Eğitim', 'Test'], loc='lower right')
plt.show()
#hata grafiği
plt.plot(history.history['loss'],"--")
plt.plot(history.history['val_loss'])
plt.title('Model Hataları')
plt.ylabel('Hatalar')
plt.xlabel('epoch (devir)')
plt.legend(['Eğitim', 'Test'], loc='upper right')
plt.show()

#model dosyası oluşturma
#model.save('bc-model.h5')
