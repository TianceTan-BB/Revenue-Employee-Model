import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# CSV:nyc_disco_enriched.csv, NY_audit__0911_classification_v3.csv, employeedata2.csv, Revenue_result.csv from revenue model

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

merge2['Company size'].value_counts()

#-----------------------------------------Data Preprocessing-----------------------------------------------------------


dfff=pd.read_csv('employeedata2.csv')

# Extract employee numbers from clay data
mask_domain = dfff['Num of Emp (Domain)'].astype(str).str.contains('employees')
dfff.loc[~mask_domain, 'Num of Emp (Domain)'] = np.nan

for col in ['Num of Emp (Domain)']:
    dfff[col] = dfff[col].where(~dfff[col].astype(str).str.contains(r'\b(198[0-9]|199[0-9]|200[0-9]|201[0-9]|202[0-3])\b'), '')

dfff['num_employees'] = dfff['Num of Emp (Domain)'].str.extract(r'([\d,]+[+]?)')[0]
dfff['num_employees_cleaned'] = dfff['num_employees'].str.replace(',', '').str.replace('+', '')


dfff=dfff.drop_duplicates('UniqueID')

# Create number of employees classification column
dfff['num_employees_cleaned'] = pd.to_numeric(dfff['num_employees_cleaned'], errors='coerce')
bins = [-1, 10, 50, 300, float('inf')]
labels = ['0-10', '10-50', '50-300', '300+']
dfff['Company size2'] = pd.cut(dfff['num_employees_cleaned'], bins=bins, labels=labels, right=False)

dfff['Company size2'].value_counts()


dfff=dfff[['UniqueID','Company size2']]

merge3=pd.merge(merge2,dfff,on='UniqueID',how='left')
merge3['Company size']=merge3['Company size'].fillna(merge3['Company size2'])
merge3=merge3.drop(columns='Company size2')


#-----------------------------------------Feature Engineering-----------------------------------------------------------

df=merge3.copy()

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

df=df.reset_index()
df['Founded Year'].fillna('Unknown', inplace=True)

df_rev=pd.read_csv('Revenue_result.csv')

df=pd.merge(df,df_rev,left_on='UniqueID',right_on='BusinessID')

df=df.drop(columns=['index','Revenue'])

#--------------------------------------------Model--------------------------------------------------------

business_ids = df['UniqueID']
df.drop('UniqueID', axis=1, inplace=True)

# Split train and predict data sets
train_df = df[df['Company size'].notna()]
predict_df = df[df['Company size'].isna()]

# One-hot encoding for categorical columns
categorical_features = ['Residential', 'Commercial', 'Founded Year', 'category', 'Clearbit_Revenue']
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(train_df[categorical_features])
encoded_predict_features = encoder.transform(predict_df[categorical_features])

# Standardizing the 'score' column
scaler = StandardScaler()
score_train = scaler.fit_transform(train_df[['score']])
score_predict = scaler.transform(predict_df[['score']])

# Concatenate 'score' with one-hot encoded features
X_train = pd.concat([pd.DataFrame(score_train), pd.DataFrame(encoded_features.toarray())], axis=1)
X_predict = pd.concat([pd.DataFrame(score_predict), pd.DataFrame(encoded_predict_features.toarray())], axis=1)

y_train = train_df['Company size']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Model evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predicting missing 'Company size'
predicted_sizes = clf.predict(X_predict)
predict_indices = predict_df.index
df.loc[predict_indices, 'Company size'] = predicted_sizes

df['Company size'].value_counts()

df=df[['BusinessID','Company size','Clearbit_Revenue']]

df.to_csv('result.csv')