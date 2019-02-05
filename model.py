import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
#import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataSet.csv')
#df.info()
df.replace('male', 0, inplace = True)
df.replace('female', 1, inplace = True)
X = df.drop('label', axis = 1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #increase random_state

rfc = RandomForestClassifier(n_estimators = 50)
print("Training model...")
rfc.fit(X_train, y_train)
scores = cross_val_score(rfc, X, y, cv = 50)
print("Model trained.")
print('Acuracy: ', scores.mean()*100, '%')

#subprocess.call(["Rscript wav_analyzer.r"])
print("Measuring acoustic parameters...")
os.system("Rscript wav_analyzer.r >/dev/null 2>&1")
print("   Done.")


df_pred = pd.read_csv('testData.csv')
#df_pred.info()
df_names = df_pred['sound.files'].copy()
df_pred.drop(['sound.files', 'duration', 'selec', 'peakf', 'Unnamed: 0'], axis=1, inplace=True) #inplace=True
#df_pred.info()
#print(df_pred.head())
prediction = rfc.predict(df_pred)
#print(prediction)
#if prediction[0] == 0:
 #   print("The subject is a male")
#else:
 #   print("The subject is a female")

#print(df_names.head())

for i in range(len(prediction)):
    #print(df_names.at[i,'sound.files'], end=':\n')
    print(df_names[i].split('.')[0], " is a ", end='')
    if prediction[i] == 0:
        print("male")
    else:
        print("female")
