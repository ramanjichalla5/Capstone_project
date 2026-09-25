# Data Pipeline

Run `python data_pipeline/main.py`. It scrapes at least 60 books from five catalogue pages, cleans price/rating/availability, computes `price_inr = price_gbp * 105.50`, and creates `books.db` with `categories` and `books` tables. Unexpected numeric values are median-imputed; rows without a usable title/category are dropped.

The script executes and prints six saved SQL query results covering `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `IN`, `BETWEEN`, and a category join. It then reads two results with `pd.read_sql` and compares the join with an in-memory `pd.merge` result.