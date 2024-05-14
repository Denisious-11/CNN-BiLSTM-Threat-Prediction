import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

lf =pd.read_csv('Data/IoT Network Intrusion Dataset.csv')
print(lf)

data = lf.replace([np.inf, -np.inf], np.nan) 
data = data.dropna()

data.drop(["Src_IP","Dst_IP","Src_Port","Dst_Port","Timestamp","Protocol","Flow_ID"],axis=1,inplace=True)

print(data['Label'].value_counts())

print(data['Cat'].value_counts())

print(data['Sub_Cat'].value_counts())


print("DataFrame Information:")
print(data.info())

print("\nDataFrame Summary Statistics:")
print(data.describe())

print("\nFirst Few Rows of the DataFrame:")
print(data.head())

plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
data['Label'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
data['Cat'].value_counts().plot(kind='bar', color='salmon')
plt.title('Distribution of Categories')
plt.xlabel('Cat')
plt.ylabel('Count')


plt.subplot(2, 2, 3)
data['Sub_Cat'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Distribution of Sub-Categories')
plt.xlabel('Sub_Cat')
plt.ylabel('Count')

plt.tight_layout()
plt.show()


set1 = {'Anomaly': 1, 'Normal': 0}

data['Label'] = data['Label'].map(lambda x: set1[x])
print(data['Label'].value_counts())

set2 = {'Mirai': 1, 'Scan': 2,'DoS': 3, 'MITM ARP Spoofing': 4,'Normal' : 0}

data['Cat'] = data['Cat'].map(lambda x: set2[x])
print(data['Cat'].value_counts())

set3 =        { 'Mirai-UDP Flooding' : 1, 'Mirai-Hostbruteforceg' : 2,'DoS-Synflooding' : 3, 'Mirai-HTTP Flooding' : 4,'Mirai-Ackflooding' : 5,'Scan Port OS' : 6,'MITM ARP Spoofing' : 7, 'Scan Hostport' : 8,'Normal' : 0}

data['Sub_Cat'] = data['Sub_Cat'].map(lambda x: set3[x])
print(data['Sub_Cat'].value_counts())


data.to_csv("Data/pre_dataset.csv", index = False)