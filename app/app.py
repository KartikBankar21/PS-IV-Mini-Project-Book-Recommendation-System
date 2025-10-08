# app/app.py
from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import difflib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'recommender.pkl')

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model payload at startup
payload = joblib.load(MODEL_PATH)
model = payload['model']
scaler = payload['scaler']
features_columns = payload['features_columns']
df = payload['df']   # pandas DataFrame
title_to_index = payload['title_to_index']

# Simple fuzzy search helper (substring & difflib fallback)
def find_title(query, max_results=10):
    q = query.lower().strip()
    matches = df[df['title'].str.lower().str.contains(q, na=False)]
    if len(matches) == 0:
        # fallback to difflib closest matches on title list
        titles = df['title'].tolist()
        close = difflib.get_close_matches(query, titles, n=max_results, cutoff=0.4)
        return df[df['title'].isin(close)]
    return matches.head(max_results)

def recommend(title, n=5):
    if title not in title_to_index:
        return []
    idx = title_to_index[title]
    # Build features again from df for kneighbors input
    # (we saved columns names earlier - rebuild features for query dataset)
    # We'll recreate the features row matrix used during training:
    # for speed in production, save the features matrix too.
    features_rows = []
    # recreate features_df equivalently
    import pandas as pd
    rating_bucket = lambda x: '0-1' if x <=1 else ('1-2' if x<=2 else ('2-3' if x<=3 else ('3-4' if x<=4 else '4-5')))
    rating_buckets = df['average_rating'].apply(rating_bucket)
    rating_df = pd.get_dummies(rating_buckets)
    language_df = pd.get_dummies(df['language_code'].fillna('unknown'))
    features_df = pd.concat([rating_df, language_df, df[['average_rating','ratings_count','num_pages']].reset_index(drop=True)], axis=1).reindex(columns=features_columns, fill_value=0)
    features_scaled = scaler.transform(features_df)
    neighs = model.kneighbors(features_scaled, n_neighbors=n+1, return_distance=False)
    rec_ids = neighs[idx]
    recs = [df.loc[i].to_dict() for i in rec_ids if i != idx]
    return recs

@app.route('/')
def index():
    # show hero + some top-rated samples
    top10 = df.sort_values(by='average_rating', ascending=False).head(12)
    return render_template('index.html', top10=top10.to_dict(orient='records'))

@app.route('/search')
def search():
    q = request.args.get('q', '')
    if not q:
        return redirect(url_for('index'))
    results = find_title(q)
    return render_template('results.html', query=q, results=results.to_dict(orient='records'))

@app.route('/book')
def book_detail():
    title = request.args.get('title')
    if not title:
        return redirect(url_for('index'))
    if title not in title_to_index:
        # try fuzzy fallback
        matches = find_title(title, max_results=1)
        if matches.shape[0]==0:
            return render_template('book.html', book=None, recs=[])
        title = matches.iloc[0]['title']
    book_row = df[df['title'] == title].iloc[0].to_dict()
    recs = recommend(title, n=5)
    return render_template('book.html', book=book_row, recs=recs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
