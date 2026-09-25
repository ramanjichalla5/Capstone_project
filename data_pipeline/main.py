from pathlib import Path
import json, re, sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
RATE = 105.50
BASE = "https://books.toscrape.com/catalogue/page-{}.html"

def scrape():
    rows = []
    for page in range(1, 6):
        soup = BeautifulSoup(requests.get(BASE.format(page), timeout=30).text, "html.parser")
        for card in soup.select("article.product_pod"):
            rating = card.select_one("p.star-rating")
            detail = BeautifulSoup(requests.get("https://books.toscrape.com/catalogue/" + card.h3.a["href"].split("catalogue/")[-1], timeout=30).text, "html.parser")
            crumbs = detail.select("ul.breadcrumb li")
            category = crumbs[-2].get_text(strip=True) if len(crumbs) > 2 else "Unknown"
            rows.append({"title": card.h3.a["title"], "price": card.select_one(".price_color").get_text(strip=True),
                         "star_rating": next((x for x in rating.get("class", [])[1:]), ""),
                         "availability": card.select_one(".availability").get_text(" ", strip=True), "category": category})
    return pd.DataFrame(rows)

def main():
    df = scrape().drop_duplicates("title")
    df["price_gbp"] = pd.to_numeric(df["price"].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    df["rating"] = df["star_rating"].str.title().map({"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5})
    df["in_stock"] = df["availability"].str.contains("In stock", case=False, na=False)
    for col in ("price_gbp", "rating"):
        df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=["title", "category"])
    df["price_inr"] = (df["price_gbp"] * RATE).round(2)
    df = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].head(100)
    db = ROOT / "books.db"
    if db.exists(): db.unlink()
    with sqlite3.connect(db) as con:
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
        cats = pd.DataFrame({"category_name": df.category.unique()}); cats.to_sql("categories", con, if_exists="append", index=False)
        ids = pd.read_sql("SELECT * FROM categories", con)
        out = df.merge(ids, left_on="category", right_on="category_name").drop(columns=["category", "category_name"])
        out.to_sql("books", con, if_exists="append", index=False)
        queries = {
            "where": "SELECT * FROM books WHERE in_stock = 1",
            "order_limit": "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10",
            "distinct": "SELECT DISTINCT rating FROM books ORDER BY rating",
            "in": "SELECT title, rating FROM books WHERE rating IN (4, 5)",
            "between": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30",
            "join": "SELECT b.title, c.category_name, b.rating FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.rating DESC LIMIT 10",
        }
        results = {name: pd.read_sql(sql, con).to_dict("records") for name, sql in queries.items()}
        (ROOT / "query_outputs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        print("pd.read_sql rows:", len(pd.read_sql(queries["where"], con)), len(pd.read_sql(queries["join"], con)))
        sql_join = pd.read_sql(queries["join"], con)
        merged = df.merge(ids, left_on="category", right_on="category_name").rename(columns={"title": "title"})
        merged = merged.sort_values("rating", ascending=False).head(10)[["title", "category_name", "rating"]].reset_index(drop=True)
        print("SQL/pandas merge equivalent:", sql_join.reset_index(drop=True).equals(merged))

if __name__ == "__main__": main()