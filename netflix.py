# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set graph style
sns.set_theme(style="whitegrid")

print("Libraries imported successfully!")



"""Read dataset"""

df = pd.read_csv("netflix_titles.csv")

print("Dataset loaded successfully!")

"""display first five row"""

df.head()

"""display last five row"""

df.tail()

"""check total rows & columns"""

print("Dataset Shape:", df.shape)

"""check column name"""

print(df.columns)

df.info()

df.describe()

df.describe(include="object")

df.isnull().sum()

missing = (df.isnull().sum() / len(df)) * 100

print(missing.sort_values(ascending=False))

print("Duplicate Rows:", df.duplicated().sum())

df_clean = df.copy()

columns = ["director", "cast", "country", "rating", "duration"]

for col in columns:
    df_clean[col] = df_clean[col].fillna("Unknown")

df_clean["date_added"] = df_clean["date_added"].fillna("Unknown")

print(df_clean.isnull().sum())

"""Data transformation"""

df_clean["date_added"] = pd.to_datetime(
    df_clean["date_added"].str.strip(),
    format="%B %d, %Y",
    errors="coerce"
)

df_clean["date_added"].head()

df_clean["year_added"] = df_clean["date_added"].dt.year

df_clean["month_added"] = df_clean["date_added"].dt.month

"""Univariate Analysis"""

print(df_clean["type"].value_counts())

sns.countplot(data=df_clean, x="type")

plt.title("Movies vs TV Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12,5))

df_clean["release_year"].value_counts().sort_index().plot()

plt.title("Content Released by Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.show()

plt.figure(figsize=(10,5))

df_clean["rating"].value_counts().head(10).plot(
    kind="bar"
)

plt.title("Top 10 Content Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

"""Bivariate Analysis"""

plt.figure(figsize=(12,5))

sns.countplot(
    data=df_clean,
    x="release_year",
    hue="type",
    order=df_clean["release_year"].value_counts().head(10).index
)

plt.title("Movies and TV Shows by Release Year")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,5))

sns.countplot(
    data=df_clean,
    x="release_year",
    hue="type",
    order=df_clean["release_year"].value_counts().head(10).index
)

plt.title("Movies and TV Shows by Release Year")
plt.xticks(rotation=45)
plt.show()

"""Genre Analysis"""

genres = df_clean["listed_in"].str.split(", ").explode()

top_genres = genres.value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_genres.values,
    y=top_genres.index
)

plt.title("Top 10 Netflix Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.show()

"""Country Analysis"""

countries = df_clean["country"]

countries = countries[countries != "Unknown"]

top_countries = countries.str.split(", ").explode().value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_countries.values,
    y=top_countries.index
)

plt.title("Top 10 Countries by Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.show()

"""Movie Duration Analysis"""

movies = df_clean[df_clean["type"] == "Movie"].copy()

movies["duration_minutes"] = pd.to_numeric(
    movies["duration"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

plt.figure(figsize=(10,5))

sns.histplot(
    movies["duration_minutes"].dropna(),
    bins=30,
    kde=True
)

plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")
plt.show()

print("TOTAL TITLES:", len(df_clean))

print("\nCONTENT TYPES:")
print(df_clean["type"].value_counts())

print("\nTOP 5 RELEASE YEARS:")
print(df_clean["release_year"].value_counts().head(5))

print("\nTOP 5 GENRES:")
print(genres.value_counts().head(5))

print("\nTOP 5 COUNTRIES:")
print(top_countries.head(5))

print("\nMISSING VALUES:")
print(df.isnull().sum())

"""In this project, I performed Exploratory Data Analysis on the Netflix Movies and TV Shows dataset using Python. I used Pandas and NumPy for data analysis and Matplotlib and Seaborn for data visualization. I explored content types, release years, genres, countries, ratings and movie durations. I also checked missing values and duplicate records. The analysis helped me understand the distribution and characteristics of Netflix content."""