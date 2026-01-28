# ============================================================
# Hybrid MF + Fuzzy Recommender System (Single File)
# Author: Pratyush Ranjan Mohanty
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score

import skfuzzy as fuzz
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------
# 0. Reproducibility
# ------------------------------
np.random.seed(42)

# ------------------------------
# 1. Dataset Path
# ------------------------------
DATA_PATH = "appliances_with_sentiment.parquet"   # change if needed
OUTPUT_DIR = "experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# 2. Load & Prepare Data
# ------------------------------
data = pd.read_parquet(DATA_PATH).dropna(subset=[
    "reviewerID", "asin", "overall", "SentimentScore", "SentimentConfidence"
])

# Rating & Sentiment normalization (CORRECT)
data["RatingNorm"] = (data["overall"] / 5.0).clip(0, 1)
data["SentimentNorm"] = (data["SentimentScore"] / 5.0).clip(0, 1)

assert data["RatingNorm"].between(0, 1).all()
assert data["SentimentNorm"].between(0, 1).all()
assert data["SentimentConfidence"].between(0, 1).all()

# Train–Validation split
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

# User & Item mappings
user_map = {u: i for i, u in enumerate(train_df["reviewerID"].unique())}
item_map = {i: j for j, i in enumerate(train_df["asin"].unique())}

def map_ids(df):
    df = df.copy()
    df["u"] = df["reviewerID"].map(user_map)
    df["i"] = df["asin"].map(item_map)
    return df.dropna(subset=["u", "i"]).astype({"u": int, "i": int})

train = map_ids(train_df)
val = map_ids(val_df)

n_users, n_items = len(user_map), len(item_map)

R_train = csr_matrix((train["RatingNorm"], (train["u"], train["i"])),
                     shape=(n_users, n_items))
S_train = csr_matrix((train["SentimentNorm"], (train["u"], train["i"])),
                     shape=(n_users, n_items))

# ------------------------------
# 3. Matrix Factorization (Adam)
# ------------------------------
class MFAdam:
    def __init__(self, R, n_factors, lr, reg, epochs, decay):
        self.R = R
        self.n_users, self.n_items = R.shape
        self.k, self.lr0 = n_factors, lr
        self.reg, self.epochs, self.decay = reg, epochs, decay

        self.U = np.random.normal(0, 0.1, (self.n_users, self.k))
        self.V = np.random.normal(0, 0.1, (self.n_items, self.k))
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.mu = np.mean(R.data)

        self.users, self.items = R.nonzero()
        self.ratings = R.data

    def predict(self, u, i):
        pred = self.mu + self.bu[u] + self.bi[i] + self.U[u] @ self.V[i]
        return np.clip(pred, 0, 1)

    def fit(self, val_df, col):
        for epoch in range(self.epochs):
            for u, i, r in zip(self.users, self.items, self.ratings):
                pred = self.predict(u, i)
                e = r - pred
                self.bu[u] += self.lr0 * (e - self.reg * self.bu[u])
                self.bi[i] += self.lr0 * (e - self.reg * self.bi[i])
                self.U[u] += self.lr0 * (e * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr0 * (e * self.U[u] - self.reg * self.V[i])

# Hyperparameters (example – tune separately)
mf_ratings = MFAdam(R_train, n_factors=64, lr=0.002, reg=0.001, epochs=40, decay=0)
mf_sentiment = MFAdam(S_train, n_factors=32, lr=0.002, reg=0.001, epochs=40, decay=0)

mf_ratings.fit(val, "RatingNorm")
mf_sentiment.fit(val, "SentimentNorm")

# ------------------------------
# 4. Fuzzy Inference System
# ------------------------------
x = np.linspace(0, 1, 101)
labels = ["very_low", "low", "medium", "high", "very_high"]
centers = [0.2, 0.4, 0.6, 0.8, 1.0]
sigma = 0.12

rating = Antecedent(x, "rating")
sentiment = Antecedent(x, "sentiment")
confidence = Antecedent(x, "confidence")
recommendation = Consequent(x, "recommendation")

for lbl, c in zip(labels, centers):
    rating[lbl] = fuzz.gaussmf(x, c, sigma)
    sentiment[lbl] = fuzz.gaussmf(x, c, sigma)
    confidence[lbl] = fuzz.gaussmf(x, c, sigma)
    recommendation[lbl] = fuzz.gaussmf(x, c, sigma)

def get_label(v):
    if v <= 0.30: return "very_low"
    if v <= 0.50: return "low"
    if v <= 0.70: return "medium"
    if v <= 0.90: return "high"
    return "very_high"

# Rule learning
buckets = defaultdict(list)
for _, row in train.iterrows():
    r = mf_ratings.predict(row.u, row.i)
    s = mf_sentiment.predict(row.u, row.i)
    c = row.SentimentConfidence
    buckets[(get_label(r), get_label(s), get_label(c))].append(row.RatingNorm)

rules = []
for (rL, sL, cL), vals in buckets.items():
    if len(vals) >= 5:
        outL = get_label(np.median(vals))
        rules.append(Rule(rating[rL] & sentiment[sL] & confidence[cL],
                          recommendation[outL]))

rules.append(Rule(rating["medium"] & sentiment["medium"] & confidence["medium"],
                  recommendation["medium"]))

fuzzy_system = ControlSystem(rules)

# ------------------------------
# 5. Hybrid Predictor
# ------------------------------
class HybridRecommender:
    def __init__(self, rm, sm, fis, alpha=0.45, beta=0.28):
        self.rm, self.sm, self.fis = rm, sm, fis
        self.alpha, self.beta = alpha, beta

    def predict(self, u, i, conf):
        r = self.rm.predict(u, i)
        s = self.sm.predict(u, i)
        sim = ControlSystemSimulation(self.fis, clip_to_bounds=True)
        sim.input["rating"] = r
        sim.input["sentiment"] = s
        sim.input["confidence"] = conf
        sim.compute()
        f = np.clip(sim.output["recommendation"] + self.beta, 0, 1)
        return np.clip(self.alpha * f + (1 - self.alpha) * r, 0, 1)

hybrid = HybridRecommender(mf_ratings, mf_sentiment, fuzzy_system)

# ------------------------------
# 6. Evaluation
# ------------------------------
def eval_model(name, preds, truth):
    rmse = np.sqrt(mean_squared_error(truth, preds))
    mae = mean_absolute_error(truth, preds)
    print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}")

truth = val["RatingNorm"].values * 5
preds_hybrid = [hybrid.predict(u, i, c) * 5 for u, i, c in zip(val.u, val.i, val.SentimentConfidence)]
preds_mf = [mf_ratings.predict(u, i) * 5 for u, i in zip(val.u, val.i)]

eval_model("MF Only", preds_mf, truth)
eval_model("Hybrid MF + FIS", preds_hybrid, truth)

# ------------------------------
# 7. Save Models
# ------------------------------
joblib.dump(mf_ratings, os.path.join(OUTPUT_DIR, "mf_ratings.pkl"))
joblib.dump(mf_sentiment, os.path.join(OUTPUT_DIR, "mf_sentiment.pkl"))
joblib.dump(fuzzy_system, os.path.join(OUTPUT_DIR, "fuzzy_system.pkl"))

print("Pipeline finished successfully.")
