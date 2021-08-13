import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


algorithms = {
    # # ランダムフォレスト
    # "RandomForest": {
    #     "model": RandomForestClassifier,
    #     "param": {
    #         'max_depth': ("int", 2, 6),
    #         'n_estimators': ("int", 10, 1000),
    #         "class_weight": 'balanced',
    #         "random_state": 42
    #     },
    #     "seed": 42
    # },

    # XGBoost
    "xgboost": {
        "model": xgb.XGBClassifier,
        "param": {
            'objective': 'binary:logistic',
            'max_depth': ("int", 2, 6),
            'n_estimators': ("int", 10, 1000),
            'learning_rate': ("float", 1e-8, 1.0),
            "eval_metric": 'mlogloss',
            "use_label_encoder": False,
            "nthread": 1,
            "seed": 42
        },
        "seed": 42
    },

    # ロジスティック回帰
    "LogisticRegression": {
        "model": LogisticRegression,
        "param": {
            'C': ("float", 1e-5, 1.00),
            # 'max_iter': ("int", 10, 500),
            "solver": ("categorical", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
            "random_state": 42
        },
        "seed": 42
    },

    # サポートベクターマシン
    "SVC": {
        "model": SVC,
        "param": {
            "probability": True,
            'C': ("float", 1e-5, 1.00),
            # 'max_iter': ("int", 10, 500),
            "kernel": ("categorical", ["linear", "rbf", "sigmoid", "poly"]),
            "random_state": 42
        },
        "seed": 42
    },

    # MLP
    "MLP": {
        "model": MLPClassifier,
        "param": {
            "activation": ("categorical", ["identity", "logistic", "tanh", "relu"]),
            "solver": ("categorical", ["sgd", "adam"]),
            "learning_rate": ("categorical", ["constant", "invscaling", "adaptive"]),
            "momentum": ("float", 1e-4, 1.00),
            "random_state": 42
        },
        "seed": 42
    }
}
