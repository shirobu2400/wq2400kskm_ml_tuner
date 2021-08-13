# wq2400kskm_ml_tuner

# 概要
ハイパーパラメータを行うツール（自分用）
kaggle とかで使う

# 操作方法
```bash
# チューニング
$ python main.py --tune
```

## オプション
- tune
    - ハイパーパラメータチューニング
- test
    - バリデーションしてテスト
- predict
    - 問題に対して予測
    - predict_proba
        - 確率で出力
