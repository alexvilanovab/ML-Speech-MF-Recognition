import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataSet.csv')
df.info()
df.replace('male', 0, inplace = True)
df.replace('female', 1, inplace = True)
X = df.drop('label', axis = 1)
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #increase random_state

rfc = RandomForestClassifier(n_estimators = 50)
rfc.fit(X_train, y_train)
scores = cross_val_score(rfc, X, y, cv = 50)
print('Acuracy: ', scores.mean()*100, '%')

#Create a heat map of the ecovariance matrix
#plt.figure(figsize=(14,12))

#sns.heatmap(df.corr(), linewidths=0.1, annot = True)
#plt.show()

prediction = rfc.predict( [[0.1984445, 0.06684052, 0.2157356, 0.1375283, 0.264536, 0.1270077, 3.38914, 20.50335, 0.8929154, 0.3376926, 0.1200362, 0.1984445, 0.1396227, 0.04349112, 0.2791139, 0.4190832, 0, 5.081836, 5.081836, 0.07727807]] )

if prediction[0] == 0:
    print("The subject is a male")
else:
    print("The subject is a female")
