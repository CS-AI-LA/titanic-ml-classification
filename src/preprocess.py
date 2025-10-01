# src/preprocess.py
"""
Simple Titanic preprocessing:
- extracts Title from Name
- fills Age by median per Title+Pclass where possible
- fills Embarked and Fare
- converts Sex to numeric
- one-hot encodes Embarked and Title
Saves processed dataframe to data/processed_train.csv
"""

import os
import pandas as pd

def load_data(path='data/train.csv'):
    return pd.read_csv(path)

def extract_title(df):
    df['Title'] = df['Name'].str.extract(r",\s*([^\.]+)\.", expand=False)

    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles,'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    return df

def fill_age(df):
    df['Age'] = df.groupby(['Title','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    return df

def preprocess(df):
    df = extract_title(df)
    df = fill_age(df)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    kep_cols = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']
    df = df[kep_cols]

    df['Sex'] = df['Sex'].map({'male':0,'female':1}).astype(int)

    df = pd.get_dummies(df,columns=['Embarked','Title'],drop_first=True)

    return df

def main():
    os.makedirs('data',exist_ok=True)
    df = load_data()
    df_proc = preprocess(df)
    df_proc.to_csv('data/processed_train.csv',index=False)
    print("Saved processed data to data/processed_train.csv")
    print('Columns:',df_proc.columns.tolist())
    print('Shape:',df_proc.shape)



if __name__ == '__main__':
    main()

