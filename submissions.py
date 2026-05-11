from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "train.csv"
TEST_PAIRS_PATH = HERE / "test_pairs.csv"
OUT_PATH = HERE / "submission.csv"


def predict(train: pd.DataFrame, test_pairs: pd.DataFrame) -> pd.DataFrame:
    unique_users = train['user_id'].unique()
    unique_items = train['item_id'].unique()
    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {i: j for j, i in enumerate(unique_items)}
    n_users, n_items = len(user_map), len(item_map)
    mu = train['rating'].mean()

    movies = pd.read_csv('movies.csv')
    item_to_genre = movies.set_index('item_id')['genres'].to_dict()
    train_with_genres = train.merge(movies, on='item_id')
    genre_means = train_with_genres.groupby('genres')['rating'].mean().to_dict()
    K = 20
    alpha = 0.005
    beta = 0.02
    epochs = 27
    

    P = np.random.normal(0, 0.1, (n_users, K))
    Q = np.random.normal(0, 0.1, (n_items, K))
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)


    train_list = []
    for r in train.itertuples():
        train_list.append((user_map[r.user_id], item_map[r.item_id], r.rating))

    print(f"Training Funk SVD... (Epochs: {epochs})")
    for epoch in range(epochs):
        np.random.shuffle(train_list)
        for u, i, r_ui in train_list:

            prediction = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
            error = r_ui - prediction
            bu[u] += alpha * (error - beta * bu[u])
            bi[i] += alpha * (error - beta * bi[i])
            old_p_u = P[u].copy()
            P[u] += alpha * (error * Q[i] - beta * P[u])
            Q[i] += alpha * (error * old_p_u - beta * Q[i])
            

    final_preds = []
    for row in test_pairs.itertuples():
        u_id, i_id = row.user_id, row.item_id
        
        if u_id in user_map and i_id in item_map:
            u_idx, i_idx = user_map[u_id], item_map[i_id]
            res = mu + bu[u_idx] + bi[i_idx] + np.dot(P[u_idx], Q[i_idx])
            final_preds.append(res)
        else:
            genre = item_to_genre.get(i_id)
            final_preds.append(genre_means.get(genre, mu))

    out = test_pairs[["user_id", "item_id"]].copy()
    out["predicted_rating"] = np.clip(final_preds, 1.0, 5.0)
    
    return out


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test_pairs = pd.read_csv(TEST_PAIRS_PATH)
    preds = predict(train, test_pairs)
    preds.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(preds):,} predictions to {OUT_PATH.name}")


if __name__ == "__main__":
    main()
