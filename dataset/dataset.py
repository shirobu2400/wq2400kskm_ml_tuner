import pandas as pd
import os

label_column = "Survived"

def train_load(path="./dataset/"):
    df = pd.read_csv(os.path.join(path, "train.csv"))
    df = df.set_index("PassengerId")
    df_x = df.drop([label_column], axis=1)[["Pclass", "Age", "Sex"]]
    df_x["Sex"] = pd.get_dummies(df_x["Sex"])
    df_y = df[label_column]
    return df_x.fillna(0), df_y, df.index

def test_load(path="./dataset/"):
    df = pd.read_csv(os.path.join(path, "test.csv"))
    df = df.set_index("PassengerId")
    df_x = df[["Pclass", "Age", "Sex"]]
    df_x["Sex"] = pd.get_dummies(df_x["Sex"])
    return df_x.fillna(0), df.index, label_column
