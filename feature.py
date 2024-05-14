import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from imblearn.over_sampling import SMOTE
import pickle

file_path = 'Data/pre_dataset.csv'
df = pd.read_csv(file_path)

X = df.drop(columns=['Cat']).values  
y = df['Cat']

print("Class Counts Before Balancing:")
print(y.value_counts())

smote = SMOTE()
X_resampled, y_resampled = smote.fit_sample(X, y)

print("Data Balancing")

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_feature_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=4242, max_iter=50, perc=90)#70
boruta_feature_selector.fit(X_resampled, y_resampled)

X_selected = boruta_feature_selector.transform(X_resampled)

selected_features_mask = boruta_feature_selector.support_

selected_feature_names = [col for col, selected in zip(df.columns, selected_features_mask) if selected]

selected_feature_names.append('Cat')

df_selected_features = pd.DataFrame(X_selected, columns=selected_feature_names[:-1])
df_selected_features['Cat'] = y_resampled

boruta_pickle_file_path = 'Extra/boruta_feature_selector.pkl'
with open(boruta_pickle_file_path, 'wb') as pickle_file:
    pickle.dump(boruta_feature_selector, pickle_file)

selected_file_path = 'Data/final_dataset.csv'
df_selected_features.to_csv(selected_file_path, index=False)

print("\nSelected and Balanced Dataset Shape:", df_selected_features.shape)
print(df_selected_features.head(10))
