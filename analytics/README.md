# Analytics Pipeline

Run `python analytics/main.py`. It loads `sns.load_dataset('titanic')` once, immediately writes `titanic.csv`, cleans using the missingness threshold, prints profiling, outlier/skew/correlation analysis, creates the EDA charts, and builds the three classifier pipelines on one stratified split. Preprocessing is inside each `ColumnTransformer`, so imputers, one-hot encoding, and scaling fit only on training data.

The script also compares baseline/class-weighted/SMOTE models, tunes an OOB Random Forest, runs fare regression with MAE/RMSE/R2/adjusted-R2 and residual analysis, writes `model_comparison.csv`, and saves/reloads `best_pipeline.joblib`. `adult_male` and `alone` are excluded from the exact six-column correlation matrix.