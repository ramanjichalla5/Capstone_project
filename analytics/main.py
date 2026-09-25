from pathlib import Path
import warnings, joblib
import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT = Path(__file__).parent
warnings.filterwarnings("ignore")

def prep(features):
    numeric = [c for c in features if c in ["age", "sibsp", "parch", "fare", "pclass"]]
    categorical = [c for c in features if c in ["sex", "embarked"]]
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
                              ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])

def main():
    df = sns.load_dataset("titanic")
    df.to_csv(ROOT / "titanic.csv", index=False)
    print(df.info(), df.describe(include="all"), df.shape, sep="\n")
    missing = (df.isna().mean() * 100).loc[lambda x: x > 0]
    print("Missing percentages:\n", missing)
    for col in ["age", "fare"]: df[col] = df[col].fillna(df[col].median())
    df["embarked"] = df["embarked"].fillna("Missing")
    df = df.dropna(subset=["survived", "sex", "pclass"])
    for col in ["age", "fare"]:
        q1, q3 = df[col].quantile([.25, .75]); iqr = q3 - q1
        print(col, "IQR outliers:", int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()))
    mode = df.fare.mode().iloc[0]; print("Fare mean/median/mode:", df.fare.mean(), df.fare.median(), mode, "right-skewed")
    print("Survival by sex:\n", df.groupby("sex").survived.mean()); print("By pclass:\n", df.groupby("pclass").survived.mean()); print("By sex+pclass:\n", df.groupby(["sex", "pclass"]).survived.mean())
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]; corr = df[corr_cols].corr(); print(corr)
    pairs = [(abs(corr.loc[a, b]), a, b, corr.loc[a, b]) for i, a in enumerate(corr_cols) for b in corr_cols[i + 1:]]
    print("Strongest correlations:", sorted(pairs, reverse=True)[:2]); sns.heatmap(corr, annot=True); plt.savefig(ROOT / "correlation.png"); plt.close()
    for col in ["age", "fare"]:
        z = (df[col] - df[col].mean()) / df[col].std(); print(col, "standardized mean/std:", z.mean(), z.std())
        fig, ax = plt.subplots(1, 2); sns.histplot(df[col], ax=ax[0]); sns.boxplot(x=df[col], ax=ax[1]); fig.savefig(ROOT / (col + "_eda.png")); plt.close(fig)
    fig, ax = plt.subplots(1, 2); sns.barplot(data=df, x="sex", y="survived", ax=ax[0]); sns.barplot(data=df, x="pclass", y="survived", ax=ax[1]); fig.savefig(ROOT / "survival_story.png"); plt.close(fig)
    fig = sns.pairplot(df[["survived", "age", "fare", "pclass"]]); fig.savefig(ROOT / "pairplot.png"); plt.close("all")
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]; X, y = df[features], df.survived
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    models = {"Logistic Regression": LogisticRegression(max_iter=1000), "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42), "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)}
    metrics, fitted = [], {}
    for name, est in models.items():
        pipe = Pipeline([("preprocess", prep(features)), ("model", est)]); pipe.fit(Xtr, ytr); pred = pipe.predict(Xte); prob = pipe.predict_proba(Xte)[:, 1]
        metrics.append({"model": name, "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred), "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "auc": roc_auc_score(yte, prob)}); fitted[name] = pipe
        print(name, confusion_matrix(yte, pred)); fpr, tpr, _ = roc_curve(yte, prob); plt.plot(fpr, tpr, label=f"{name} AUC={metrics[-1]['auc']:.3f}")
        if name == "Decision Tree": plot_tree(pipe.named_steps["model"], feature_names=pipe.named_steps["preprocess"].get_feature_names_out(), class_names=["0", "1"], filled=True, max_depth=3); plt.savefig(ROOT / "decision_tree.png"); plt.close()
    plt.legend(); plt.savefig(ROOT / "roc_curves.png"); plt.close(); comparison = pd.DataFrame(metrics); print(comparison)
    for label, classifier in [("baseline", RandomForestClassifier(random_state=42)), ("balanced", RandomForestClassifier(class_weight="balanced", random_state=42))]:
        p = Pipeline([("preprocess", prep(features)), ("model", classifier)]); p.fit(Xtr, ytr); q = p.predict(Xte); print(label, precision_score(yte, q), recall_score(yte, q), f1_score(yte, q))
    smote = ImbPipeline([("preprocess", prep(features)), ("smote", SMOTE(random_state=42)), ("model", RandomForestClassifier(random_state=42))]); smote.fit(Xtr, ytr); q = smote.predict(Xte); print("SMOTE", precision_score(yte, q), recall_score(yte, q), f1_score(yte, q))
    grid = GridSearchCV(Pipeline([("preprocess", prep(features)), ("model", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42))]), {"model__n_estimators": [100, 200], "model__max_depth": [None, 6], "model__max_features": ["sqrt", "log2"]}, cv=3, scoring="f1"); grid.fit(Xtr, ytr); print("Best params:", grid.best_params_, "OOB:", grid.best_estimator_.named_steps["model"].oob_score_)
    reg_features = [c for c in features if c != "fare"]; reg = Pipeline([("preprocess", prep(reg_features)), ("model", LinearRegression())]); rtr, rte, rytr, ryte = train_test_split(df[reg_features], df.fare, test_size=.2, random_state=42); reg.fit(rtr, rytr); rp = reg.predict(rte); r2 = r2_score(ryte, rp); adj = 1 - (1-r2)*(len(ryte)-1)/(len(ryte)-len(reg.named_steps["preprocess"].get_feature_names_out())-1); print("Regression MAE/RMSE/R2/Adjusted R2:", mean_absolute_error(ryte, rp), mean_squared_error(ryte, rp)**.5, r2, adj); sns.scatterplot(x=rp, y=ryte-rp); plt.axhline(0, color="red"); plt.savefig(ROOT / "residuals.png"); plt.close()
    best = fitted["Random Forest"]; joblib.dump(best, ROOT / "best_pipeline.joblib"); loaded = joblib.load(ROOT / "best_pipeline.joblib"); print("Reload prediction:", loaded.predict(Xte.head(1)))
    comparison.to_csv(ROOT / "model_comparison.csv", index=False)

if __name__ == "__main__": main()