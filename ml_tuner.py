import numpy as np
import pandas as pd
import os
import random
import optuna
import my_algorithm
import queue
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter('ignore', FutureWarning)


class ml_tuner:
    def __init__(self, algorithms=my_algorithm.algorithms, evaluate_name="accuracy"):
        self.algorithms = algorithms
        self.evaluate_name = evaluate_name

    # 評価
    def evaluate(self, y_score, y_true):
        y_samax = np.argmax(y_score, axis=1)
        acc = accuracy_score(y_true, y_samax)
        con_mat = confusion_matrix(y_true, y_samax, labels=[0, 1])
        f1 = f1_score(y_true, y_samax)
        auc = roc_auc_score(y_true, y_score[:, 1])
        return {"accuracy": acc, "confusion_matrix": str(con_mat), "f-score": f1, "auc": auc}

    def mltune(self, model_class, parameters, seed, dataset, n_splits=5, n_trials=100):
        log_queue_columns = ["accuracy", "f-score", "auc"]
        log_queue = queue.Queue()

        # Objective関数の設定
        def objective(trial):
            random.seed(seed)
            np.random.seed(seed)

            params = {}
            for k, p in parameters.items():
                q = None
                if (type(p) is list or type(p) is tuple) and len(p) > 1:
                    if p[0] == "int":
                        q = trial.suggest_int(k, p[1], p[2])
                    if p[0] == "float":
                        q = trial.suggest_uniform(k, p[1], p[2])
                    if p[0] == "categorical":
                        q = trial.suggest_categorical(k, p[1])
                if q is None:
                    q = p
                params[k] = q

            kf = KFold(n_splits=n_splits)
            df_kfold = pd.DataFrame()
            for train_index, test_index in kf.split(dataset):
                train_dataset_x = dataset.drop(["y"], axis=1).iloc[train_index]
                train_dataset_y = dataset["y"].iloc[train_index]

                test_dataset_x = dataset.drop(["y"], axis=1).iloc[test_index]
                test_dataset_y = dataset["y"].iloc[test_index]

                model = model_class(**params)
                model.fit(train_dataset_x, train_dataset_y)

                y_pred = model.predict_proba(test_dataset_x)

                params_log = self.evaluate(y_pred, test_dataset_y)
                df = pd.DataFrame(params_log, index=["index",])
                df_kfold = pd.concat([df_kfold, df[log_queue_columns]])

            log_queue.put(df_kfold[log_queue_columns].mean())
            # df_result = pd.concat([df_result, df_kfold[["accuracy", "f-score", "auc"]].mean()])

            return params_log[self.evaluate_name]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=os.cpu_count())

        dict_result = dict(zip(log_queue_columns, [[] for _ in log_queue_columns]))
        while not log_queue.empty():
            x = log_queue.get()
            for k in log_queue_columns:
                dict_result[k].append(x[k])
        df_result = pd.DataFrame(dict_result)

        return df_result.sort_values(self.evaluate_name, ascending=False), study.best_params

    # チューニング本体
    def tune(self, x: pd.DataFrame, y: pd.Series, n_trials=100):
        result_logs = {}
        best_params = {}

        dataset = x.copy()
        dataset["y"] = y

        for name, infos in self.algorithms.items():
            print(name)
            print(infos)
            df, best_param = self.mltune(infos["model"], infos["param"], infos["seed"], dataset, n_trials=n_trials)
            result_logs[name] = df
            best_params[name] = best_param
        self.result_logs = result_logs
        self.best_params = best_params

    # テスト
    def predict(self, name: str, params: dict, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, proba: bool = False):
        algorithm = self.algorithms[name]

        seed = algorithm["seed"]
        random.seed(seed)
        np.random.seed(seed)

        for k in params.keys():
            p = algorithm["param"][k]
            if (type(p) is list) or (type(p) is tuple):
                if p[0] == "int":
                    params[k] = int(params[k])

        model = algorithm["model"](**params)
        model.fit(train_x, train_y)
        if proba:
            return model.predict_proba(test_x)
        return model.predict(test_x)

    # チューニングログの取得
    def get_log(self):
        return self.result_logs

    # 最良値を取得
    def get_best_params(self):
        return self.best_params
