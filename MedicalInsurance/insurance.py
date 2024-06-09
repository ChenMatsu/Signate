import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import G, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


if __name__ == "__main__":
    """
    1. データの読み込みと確認
    """

    print("------データ読み込みと確認------")
    df = pd.read_csv("train.csv")  # 医療保険の費用帯予測トレニングデータ
    print(df.head())
    print("------データ読み込みと確認完了------")

    """
    2. データの前処理
    """

    print("------データ前処理------")
    print("欠損値の確認: \n", df.isnull().sum())

    X = df.drop(columns=["id", "charges"])  # トレーニングデータの特徴量（説明変数）
    y = df["charges"]  # トレーニングデータの目的変数

    categorical_cols = ["sex", "smoker", "region"]
    numerical_cols = ["age", "bmi", "children"]

    # クラスデータのエンコーディングと数値データの正規化

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(), categorical_cols),
        ]
    )

    print("------データ前処理完了------")

    """
    3. データの相関を確認
    """

    print("------データ変数の相関係数確認------")

    print("数値データ相関係数の確認")
    X_corr = pd.get_dummies(X, columns=categorical_cols)
    correlation_table = X_corr.corr()
    correlation_table.to_csv("correlation.csv", index=True)
    print("共相関係数をcorrelation.csvとして保存されました.")
    # correlation_table = X.corr()
    # print(correlation_table)

    print("------データ変数の相関係数確認完了------")

    """
    4. データの可視化
    """
    print("------データの可視化------")

    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # 年齢とBMIの関係
    plt.subplot(2, 2, 1)
    sns.scatterplot(
        x="age", y="bmi", data=df, hue="charges", palette="coolwarm", alpha=0.7
    )
    plt.title("Age vs BMI")

    # 各地域の喫煙者と非喫煙者の比率
    plt.subplot(2, 2, 2)
    sns.countplot(x="region", data=df, hue="smoker", palette="coolwarm")
    plt.title("Smoker and Region")

    # 子供の数と費用の関係
    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x="children", y="charges", palette="muted", errorbar=None)
    plt.title("Average Charges by Number of Children")
    plt.xlabel("Number of Children")
    plt.ylabel("Average Charges")

    # 年齢と費用の関係
    plt.subplot(2, 2, 4)
    sns.scatterplot(
        data=df, x="age", y="charges", hue="smoker", palette="coolwarm", alpha=0.7
    )
    plt.title("Age vs Charges")
    plt.xlabel("Age")
    plt.ylabel("Charges")

    # plt.show()

    print("------データの可視化完了------")

    """
    5. モデルの構築
    """
    print("------モデルの構築------")
    model = RandomForestRegressor(random_state=0)
    # model = GradientBoostingRegressor(random_state=0)
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor())]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    try:
        best_model = joblib.load("model.pkl")
        print("------モデルの読み込み完了------")
    except Exception as e:
        print("------モデルの構築中------")

        # grid_search = GridSearchCV(
        #     estimator=pipe,
        #     param_grid=param_grid,
        #     cv=5,
        #     n_jobs=-1,
        #     verbose=2,
        #     scoring="neg_mean_squared_error",
        # )

        # grid_search.fit(X_train, y_train)

        param_grid = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__max_depth": [None, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10, 15],
            "model__min_samples_leaf": [1, 2, 4, 6],
            "model__max_features": [1.0, "sqrt", "log2"],
        }

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=100,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        print("------モデルの構築完了------")
        joblib.dump(best_model, "model.pkl")
        print("------モデルの保存完了------")

    """
    6. モデルの評価
    """
    print("------モデルの評価------")
    print("Train score: ", best_model.score(X_train, y_train))
    print("Test score: ", best_model.score(X_test, y_test))
    print("------モデルの評価完了------")

    """
    7. データの予測
    """
    print("------データの予測------")
    df_test = pd.read_csv("test.csv")
    pred = best_model.predict(df_test)
    pred = np.round(pred)
    pred = np.where(pred < 0, 0, pred)
    pred = np.where(pred > 2, 2, pred)
    pred = pred.astype(int)
    predict_df = pd.DataFrame({"id": df_test["id"], "charges": pred})
    predict_df.to_csv("predict.csv", index=False, header=False)
    # Signate 提出用
    print("予測結果をpredict.csvとして保存されました.")

    """
    8. モデル予測の可視化
    """
    df_test = pd.merge(df_test, predict_df, on="id", how="left")
    print(df_test.head())

    # 年齢と費用の関係（喫煙者別）
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_test, x="age", y="charges", hue="smoker", palette="coolwarm", alpha=0.7
    )
    plt.title("Age vs Charges (Smoker) - Test Data")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.show()
