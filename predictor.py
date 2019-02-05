#Thiss predictor uses the trained model to predict the gender of an audio recording

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

f = open("trained_classifier", "rb")
rfc = pickle.load(f)


print("Measuring acoustic parameters...")
os.system("Rscript wav_analyzer.r >/dev/null 2>&1")
print("Done.")


df_pred = pd.read_csv('testData.csv')
#Reads audio data from testData
df_names = df_pred['sound.files'].copy()
#Saves names column to display them later
df_pred.drop(['sound.files', 'duration', 'selec', 'peakf', 'Unnamed: 0'], axis=1, inplace=True) #inplace=True
#Leaves only the columns that matter to make the prediction
prediction = rfc.predict(df_pred)
#Creates an array of ints that represent the predicted gender of each row of the dataFrame

print("-"*10)
for i in range(len(prediction)):
    #Displays the predicted genders in a formatted way
    print(df_names[i].split('.')[0].title(), "is a ", end='')
    if prediction[i] == 0:
        print("male")
    else:
        print("female")
