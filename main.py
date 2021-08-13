import numpy as np
import pandas as pd
import dataset.dataset as dataset
import ml_tuner
import os
import argparse
from sklearn.model_selection import train_test_split

def main(args):
    mlt = ml_tuner.ml_tuner(evaluate_name="accuracy")
    df_train_x, df_train_y, train_index = dataset.train_load()
    df_test_x, test_index, label_column = dataset.test_load()

    # os.makedirs("data", exist_ok=True)
    # df_train_x.to_csv(os.path.join("data", "train_x.csv"))
    # df_train_y.to_csv(os.path.join("data", "train_y.csv"))
    # df_test_x.to_csv(os.path.join("data", "test_x.csv"))

    print(args)
    if args.show_table:
        print(df_train_x.head())

    if args.show_type:
        print("/* **** **** **** **** df_train_x.info() **** **** **** **** */")
        print(df_train_x.info())
        print("/* **** **** **** **** df_test_x.info() **** **** **** **** */")
        print(df_test_x.info())

    if args.tune:
        mlt.tune(df_train_x, df_train_y, n_trials=50)

        os.makedirs("best", exist_ok=True)
        best_params = mlt.get_best_params()
        for k, v in best_params.items():
            df = pd.DataFrame(v, index=["best",])
            df.to_csv(os.path.join("best", k + ".csv"))

        os.makedirs("log", exist_ok=True)
        result = mlt.get_log()
        for k, df in result.items():
            df.to_csv(os.path.join("log", k + ".csv"))

    if args.test is not None:
        model_name = args.test
        x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, random_state=0)
        df = pd.read_csv(os.path.join("best", model_name + ".csv"), index_col=0)
        params = df.loc["best"].to_dict()
        print("params", params)
        y_pred = mlt.predict(model_name, params, x_train, y_train, x_test, proba=True)
        scores = mlt.evaluate(y_pred, y_test)
        os.makedirs("score", exist_ok=True)
        pd.DataFrame(scores, index=["",]).to_csv(os.path.join("score", model_name + ".csv"))
        print(scores)

    if args.predict is not None:
        model_name = args.predict
        proba = args.predict_proba
        df = pd.read_csv(os.path.join("best", model_name + ".csv"), index_col=0)
        params = df.loc["best"].to_dict()
        print("params", params)
        pred = mlt.predict(model_name, params, df_train_x, df_train_y, df_test_x, proba=proba)
        os.makedirs("predict", exist_ok=True)
        df = pd.DataFrame({label_column: pred})
        df[test_index.name] = test_index
        df = df.set_index(test_index.name)
        df.to_csv(os.path.join("predict", model_name + ".csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_table", action="store_true")
    parser.add_argument("--show_type", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--test")
    parser.add_argument("--predict")
    parser.add_argument("--predict_proba", action="store_true")
    args = parser.parse_args()
    main(args)
