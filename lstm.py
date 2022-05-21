import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.models import Sequential
from keras.layers import LSTM,Dense,Flatten,Dropout 

def dimenstion_mod(x):
    a = list()
    for i in range(x.shape[0]):
        a.append(x[i])
    a = np.array(a) 
    return np.reshape(a, (a.shape[0], a.shape[1],1))

# df = pd.read_pickle('./data.pkl')
def lstm(df,target):
    X =  np.array(df["dim_0"].to_list())
    y = df[target].to_numpy(dtype=int)

    y = np.where(y == -1, 0, y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    X_train = dimenstion_mod(X_train)
    X_val = dimenstion_mod(X_val)
    X_test = dimenstion_mod(X_test)

    model = Sequential()
    model.add(LSTM(100,
                return_sequences=True,
                input_shape=(X_train.shape[1],1)))
    model.add(LSTM(200, return_sequences=True))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    #model.add(LSTM(20))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy)

    history = model.fit(X_train,y_train,epochs=100, validation_data=(X_val, y_val), validation_steps = 20 ,batch_size=100)
    model.save('lstm')
    y_pred = model.predict(X_test)
    y_pred = np.where(y_pred > 0.5,1,0)

    cm = metrics.confusion_matrix(y_test, y_pred[:,0])
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred[:,0])
    print(cm)

    return cm, balanced_accuracy