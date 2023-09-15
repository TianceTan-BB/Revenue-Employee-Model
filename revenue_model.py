import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# CSV:nyc_disco_enriched.csv, NY_audit__0911_classification_v3.csv, 

#-----------------------------------------Data Preparation-----------------------------------------------------------

df_clay=pd.read_csv('nyc_disco_enriched.csv')
df_clay=df_clay[['UniqueID','Revenue', 'Company size','Founded Year']]

# Create a 'completeness' column that counts non-missing values for each row
df_clay['completeness'] = df_clay.notnull().sum(axis=1)
df_clay = df_clay.sort_values(by='completeness', ascending=False)
df_clay = df_clay.drop_duplicates(subset='UniqueID', keep='first')

df_clay = df_clay.drop(columns=['completeness'])
df_clay=df_clay.reset_index()
df_clay=df_clay.drop(columns='index')
df_clay = df_clay.iloc[:-1]

df2=pd.read_csv('NY_audit__0911_classification_v3.csv')
df2=df2[['UniqueID','score','GC category', 'SubGC category','Residential', 'Commercial']].drop_duplicates()


# merge1=pd.merge(df1,df2,left_on='UniqueID',right_on='UniqueID',how='outer')
merge2=pd.merge(df2,df_clay,left_on='UniqueID',right_on='UniqueID',how='left')

#-----------------------------------------Data Preprocessing-----------------------------------------------------------
df=merge2.copy()
def categorize_year(year):
    if year < 1950:
        return 'before_1950'
    elif 1950 <= year < 1980:
        return '1950-1980'
    elif 1980 <= year < 2000:
        return '1980-2000'
    elif 2000 <= year < 2010:
        return '2000-2010'
    elif 2010 <= year < 2024:
        return '2010-2023'
    else:
        return np.nan

df['Founded Year'] = df['Founded Year'].apply(categorize_year)
df['Founded Year'].value_counts()

df = df[df['Residential'].notna()]


conditions = [
    df['GC category'].notna(),
    df['SubGC category'].notna()
]
choices = ['GC', 'SUB GC']
df['category'] = np.select(conditions, choices, default=np.nan)

df=df.drop(columns=['GC category', 'SubGC category'])

df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['score'].fillna(df['score'].mean(), inplace=True)
df['score'] = df['score'].astype(int)

df['Clearbit_Revenue']=df['Revenue']
df=df.reset_index()
df=df.drop(columns=['index','Revenue'])

df['Founded Year'].fillna('Unknown', inplace=True)
df['Company size'].fillna('Unknown', inplace=True)

#----------------------------------------Model-----------------------------------------------

business_ids = df['UniqueID']
df.drop('UniqueID', axis=1, inplace=True)

# Split the data into train and predict sets
df_train = df.dropna(subset=['Clearbit_Revenue'])
df_predict = df[df['Clearbit_Revenue'].isna()]

# Preprocess the data
df_train = pd.get_dummies(df_train, columns=['Founded Year', 'Company size', 'Residential', 'Commercial', 'category'])
df_predict = pd.get_dummies(df_predict, columns=['Founded Year', 'Company size', 'Residential', 'Commercial', 'category'])

# Ensure both dataframes have the same columns (important after one-hot encoding)
missing_cols = set(df_train.columns) - set(df_predict.columns)
for c in missing_cols:
    df_predict[c] = 0
df_predict = df_predict[df_train.columns]

X_train = df_train.drop('Clearbit_Revenue', axis=1)
y_train = df_train['Clearbit_Revenue']
X_predict = df_predict.drop('Clearbit_Revenue', axis=1)

# Scale features
scaler = StandardScaler()
X_train['score'] = scaler.fit_transform(X_train[['score']])
X_predict['score'] = scaler.transform(X_predict[['score']])

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict the missing values
predicted_values = model.predict(X_predict)
df.loc[df['Clearbit_Revenue'].isna(), 'Clearbit_Revenue'] = predicted_values

df['BusinessID'] = business_ids


df=df[['BusinessID','Company size', 'Founded Year', 'Clearbit_Revenue']]

df.replace('Unknown', np.nan, inplace=True)
df
df.to_csv('Revenue_result.csv')
