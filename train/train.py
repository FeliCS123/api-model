from unittest import TestCase
import joblib
import pandas as pd 
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import xlrd


def ingest_data(file_path : str) -> pd.DataFrame:
    return pd.read_excel(file_path)

def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    df = df[['survived', 'pclass', 'sex', 'age']]
    df.dropna(axis=0, inplace=True)
    df['sex'].replace(to_replace={'male': 0, 'female': 1}, inplace=True)
    return df


def train_model(df : pd.DataFrame) -> ClassifierMixin:
    model = KNeighborsClassifier()
    y = df["survived"]
    X = df.drop(labels='survived', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("score :", score)
    return model


df = ingest_data(r'C:\Users\felix\OneDrive\Documents\GitHub\api-model\train\titanic.xls')

df = clean_data(df)

model = train_model(df)
joblib.dump(model, "model.pkl")

from train import clean_data
class TestTrain(TestCase):
    def test_clean_data(self):
        df=pd.DataFrame({'survived ': [1,0,1], 'pclass': [1,2,3],'sex':['male','female','male'],'age': [20,30,40] })
        
        result_df=clean_data(df)

        self.assertEqual(result_df.shape[0],second=3)


        
