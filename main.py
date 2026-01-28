import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from collections import defaultdict
import skfuzzy as fuzz
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation
import optuna
import joblib

# ------------------------------
# 1) Load & Prepare Data
# ------------------------------
data = pd.read_parquet('appliances_with_sentiment.parquet').dropna(subset=['overall', 'SentimentScore', 'SentimentConfidence'])

# Normalize rating to [0,1]
data['RatingNorm'] = data['overall'] / 5.0
# Map sentiment from [-1,1] to [0,1]
data['SentimentNorm'] = (data['SentimentScore'] + 1.0) / 2.0

# Split train/val (strict separation)
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

# Mapping users/items
user_map = {u: i for i, u in enumerate(train_df['reviewerID'].unique())}
item_map = {p: i for i, p in enumerate(train_df['asin'].unique())}

def map_ids(df):
    df = df.copy()
    df['u'] = df['reviewerID'].map(user_map)
    df['i'] = df['asin'].map(item_map)
    return df.dropna(subset=['u', 'i']).astype({'u': int, 'i': int})

train = map_ids(train_df)
val = map_ids(val_df)

n_users, n_items = len(user_map), len(item_map)

# ------------------------------
# 2) Build Sparse Train Matrices
# ------------------------------
R_train = csr_matrix((train['RatingNorm'], (train['u'], train['i'])), shape=(n_users, n_items))
S_train = csr_matrix((train['SentimentNorm'], (train['u'], train['i'])), shape=(n_users, n_items))

# ------------------------------
# 3) Matrix Factorization with Corrected Adam
# ------------------------------
class MFAdam:
    def __init__(self, R, n_factors, lr, reg, n_epochs, decay):
        self.R = R
        self.n_users, self.n_items = R.shape
        self.n_factors, self.lr0 = n_factors, lr
        self.reg, self.epochs, self.decay = reg, n_epochs, decay

        self.U = np.random.normal(0, 0.1, (self.n_users, n_factors))
        self.V = np.random.normal(0, 0.1, (self.n_items, n_factors))
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.mu = np.mean(R.data) if len(R.data) > 0 else 0.0

        self.mU = np.zeros_like(self.U); self.vU = np.zeros_like(self.U)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mbu= np.zeros_like(self.bu); self.vbu= np.zeros_like(self.bu)
        self.mbi= np.zeros_like(self.bi); self.vbi= np.zeros_like(self.bi)
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

        self.users, self.items = R.nonzero()
        self.ratings = R.data

    def step(self, t):
        lr = self.lr0 / (1 + self.decay * t)
        for idx, (u, i) in enumerate(zip(self.users, self.items)):
            r = self.ratings[idx]
            pred = self.mu + self.bu[u] + self.bi[i] + np.dot(self.U[u], self.V[i])
            e = r - pred

            # Correct positive gradient updates
            g_bu = -e + self.reg * self.bu[u]
            g_bi = -e + self.reg * self.bi[i]
            g_U = -e * self.V[i] + self.reg * self.U[u]
            g_V = -e * self.U[u] + self.reg * self.V[i]

            for p, g, m, v, idx_p in [(self.bu, g_bu, self.mbu, self.vbu, u), (self.bi, g_bi, self.mbi, self.vbi, i)]:
                m[idx_p] = self.beta1 * m[idx_p] + (1-self.beta1) * g
                v[idx_p] = self.beta2 * v[idx_p] + (1-self.beta2) * (g*g)
                m_hat = m[idx_p] / (1 - self.beta1 ** (t+1))
                v_hat = v[idx_p] / (1 - self.beta2 ** (t+1))
                p[idx_p] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

            for mat, grad, m_mat, v_mat, idx_mat in [(self.U, g_U, self.mU, self.vU, u), (self.V, g_V, self.mV, self.vV, i)]:
                m_mat[idx_mat] = self.beta1 * m_mat[idx_mat] + (1-self.beta1) * grad
                v_mat[idx_mat] = self.beta2 * v_mat[idx_mat] + (1-self.beta2) * (grad*grad)
                m_hat = m_mat[idx_mat] / (1 - self.beta1 ** (t+1))
                v_hat = v_mat[idx_mat] / (1 - self.beta2 ** (t+1))
                mat[idx_mat] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def fit(self):
        for epoch in range(self.epochs):
            self.step(epoch)

    def predict(self, u, i):
        val = self.mu + self.bu[u] + self.bi[i] + np.dot(self.U[u], self.V[i])
        return np.clip(val, 0, 1)

# ------------------------------
# 4) Tune MF Models
# ------------------------------

def tune_mf(R_tr, df_map, col, trials=30):
    def objective(trial):
        params = {
            'n_factors': trial.suggest_int('n_factors', 8, 64),
            'lr': trial.suggest_float('lr', 1e-3, 1e-1, log=True),
            'reg': trial.suggest_float('reg', 1e-4, 1e-1, log=True),
            'n_epochs': trial.suggest_int('n_epochs', 10, 100),
            'decay': trial.suggest_float('decay', 1e-4, 1e-1, log=True),
        }
        model = MFAdam(R_tr, **params)
        model.fit()
        preds = np.array([model.predict(u, i) for u, i in zip(df_map['u'], df_map['i'])])
        truth = df_map[col].to_numpy()
        return np.sqrt(mean_squared_error(truth, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)
    return study.best_params

best_rating_params = tune_mf(R_train, train, 'RatingNorm', trials=30)
best_sentiment_params = tune_mf(S_train, train, 'SentimentNorm', trials=30)

mf_ratings = MFAdam(R_train, **best_rating_params); mf_ratings.fit()
mf_sentiment = MFAdam(S_train, **best_sentiment_params); mf_sentiment.fit()

joblib.dump(mf_ratings, 'mf_adamsratings.pkl')
joblib.dump(mf_sentiment, 'mf_adamssentiment.pkl')
# hybrid_recommender_fixed.py

# ------------------------------
# 5) Fuzzy Logic & Hybrid Recommender
# ------------------------------

# Fuzzy Membership Setup
# hybrid_recommender_fixed.py


# ------------------------------
# 5) Fuzzy Logic & Hybrid Recommender (Fixed Rule Threshold)
# ------------------------------

# Fuzzy Membership Setup
x = np.arange(0, 1.01, 0.01)
labels = ['very_low', 'low', 'medium', 'high', 'very_high']

rating = Antecedent(x, 'rating')
sentiment = Antecedent(x, 'sentiment')
confidence = Antecedent(x, 'confidence')
recommendation = Consequent(x, 'recommendation')

for lbl in labels:
    center = {'very_low': 0.0, 'low': 0.25, 'medium': 0.5, 'high': 0.75, 'very_high': 1.0}[lbl]
    rating[lbl] = fuzz.trapmf(x, [center-0.1, center-0.05, center+0.05, center+0.1])
    sentiment[lbl] = fuzz.gaussmf(x, center, 0.08)
    confidence[lbl] = fuzz.gaussmf(x, center, 0.08)
    recommendation[lbl] = fuzz.trapmf(x, [center-0.1, center-0.05, center+0.05, center+0.1])

def get_label(v):
    if v <= 0.15: return 'very_low'
    if v <= 0.35: return 'low'
    if v <= 0.65: return 'medium'
    if v <= 0.85: return 'high'
    return 'very_high'

# Build fuzzy rules from training predictions only
buckets = defaultdict(list)
for _, row in train.iterrows():
    rp = mf_ratings.predict(row.u, row.i)
    sp = mf_sentiment.predict(row.u, row.i)
    cp = row['SentimentConfidence']
    buckets[(get_label(rp), get_label(sp), get_label(cp))].append(row['RatingNorm'])

rules = []
for (rL, sL, cL), vals in buckets.items():
    if len(vals) >= 10:  # reduced threshold for building rules
        outL = get_label(np.median(vals))
        rules.append(Rule(rating[rL] & sentiment[sL] & confidence[cL], recommendation[outL]))

fuzzy_system = ControlSystem(rules)

# ------------------------------
# 6) Hybrid Model with FIS & Defensive Prediction
# ------------------------------
class FuzzyHybridRecommender:
    def __init__(self, rm, sm, fuzzy_sys, alpha, bias):
        self.rm, self.sm = rm, sm
        self.sys = fuzzy_sys
        self.alpha, self.bias = alpha, bias

    def predict(self, u, i, conf):
        r_pred = self.rm.predict(u, i)
        s_pred = self.sm.predict(u, i)
        sim = ControlSystemSimulation(self.sys)
        sim.input['rating'] = r_pred
        sim.input['sentiment'] = s_pred
        sim.input['confidence'] = conf
        sim.compute()

        if 'recommendation' in sim.output:
            fuzzy_out = np.clip(sim.output['recommendation'] + self.bias, 0, 1)
        else:
            fuzzy_out = r_pred  # fallback to MF prediction if fuzzy fails

        return np.clip(self.alpha * fuzzy_out + (1 - self.alpha) * r_pred, 0, 1)

# [...rest of your sections remain unchanged...]


# ------------------------------
# 7) Tune alpha and bias using Optuna
# ------------------------------
def tune_fuzzy_params(val_df, recommender_class):
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        bias = trial.suggest_float('bias_shift', -0.3, 0.3)
        model = recommender_class(mf_ratings, mf_sentiment, fuzzy_system, alpha, bias)
        preds = [model.predict(u, i, c) for u, i, c in zip(val_df.u, val_df.i, val_df.SentimentConfidence)]
        truth = val_df['RatingNorm'].values
        return np.sqrt(mean_squared_error(truth, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    return study.best_params

best_fuzzy_params = tune_fuzzy_params(val, FuzzyHybridRecommender)
hybrid_model = FuzzyHybridRecommender(mf_ratings, mf_sentiment, fuzzy_system, best_fuzzy_params['alpha'], best_fuzzy_params['bias_shift'])

# ------------------------------
# 8) Evaluation on Validation Set
# ------------------------------
def print_metrics(name, true, preds):
    def to_cls(x): return 1 if x >= 3.5 else 0
    t_cls = [to_cls(x) for x in true]
    p_cls = [to_cls(x) for x in preds]
    mse = mean_squared_error(true, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, preds)
    prec = precision_score(t_cls, p_cls, average='weighted', zero_division=0)
    rec = recall_score(t_cls, p_cls, average='weighted', zero_division=0)
    f1 = f1_score(t_cls, p_cls, average='weighted', zero_division=0)
    print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

val_preds = [hybrid_model.predict(u, i, c) * 5.0 for u, i, c in zip(val.u, val.i, val.SentimentConfidence)]
val_truth = val['RatingNorm'].values * 5.0

print_metrics("Hybrid MF + FIS", val_truth, val_preds)

# ------------------------------
# 9) Plot Membership Functions (Optional)
# ------------------------------
# Predict on validation set using only mf_ratings
val_preds_mf_ratings = [mf_ratings.predict(u, i) * 5.0 for u, i in zip(val.u, val.i)]
val_truth_ratings = val['RatingNorm'].values * 5.0

print_metrics("MF Ratings Only", val_truth_ratings, val_preds_mf_ratings)
# Predict on validation set using only mf_sentiment
val_preds_mf_sentiment = [mf_sentiment.predict(u, i) * 5.0 for u, i in zip(val.u, val.i)]
val_truth_sentiment = val['RatingNorm'].values * 5.0

print_metrics("MF Sentiment Only", val_truth_sentiment, val_preds_mf_sentiment)
val_preds_hybrid = [hybrid_model.predict(u, i, c) * 5.0 for u, i, c in zip(val.u, val.i, val.SentimentConfidence)]
val_truth_hybrid = val['RatingNorm'].values * 5.0

print_metrics("Hybrid MF + FIS", val_truth_hybrid, val_preds_hybrid)
