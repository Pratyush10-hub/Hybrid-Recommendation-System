import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ----------------------------
# 1) Load Data from JSONL
# ----------------------------
df = pd.read_json("appliances.json", lines=True)

# Keep only required columns
required_cols = ['reviewerID', 'asin', 'overall', 'reviewText']
df = df[required_cols]

# Drop invalid rows
df = df.dropna(subset=['reviewText', 'overall'])
df = df[df['overall'].between(1, 5)]

# ----------------------------
# 2) Load BERT Sentiment Model
# ----------------------------
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# 3) Sentiment Score (0–5) + Confidence (0–1)
# ----------------------------
def get_sentiment_score_and_confidence(text_list):
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    # Star scores: 1–5
    stars = torch.arange(1, 6).float().to(device)

    # Expected sentiment score in [1,5] (≈ 0–5 acceptable)
    sentiment_score = torch.sum(probs * stars, dim=1)

    # Confidence = max class probability
    sentiment_confidence = torch.max(probs, dim=1).values

    return (
        sentiment_score.cpu().numpy(),
        sentiment_confidence.cpu().numpy()
    )

# ----------------------------
# 4) Apply Sentiment Analysis (Batched)
# ----------------------------
batch_size = 16
sent_scores, confidences = [], []

for i in tqdm(range(0, len(df), batch_size), desc="Analyzing Sentiment"):
    texts = df['reviewText'].iloc[i:i+batch_size].astype(str).tolist()
    s, c = get_sentiment_score_and_confidence(texts)
    sent_scores.extend(s)
    confidences.extend(c)

df['SentimentScore'] = sent_scores                   # 0–5 (or ~1–5)
df['SentimentConfidence'] = confidences              # 0–1

# ----------------------------
# 5) Normalize Sentiment to [0,1]
# ----------------------------
df['SentimentNorm'] = (df['SentimentScore'] / 5.0).clip(0, 1)

# ----------------------------
# 6) Save to Parquet
# ----------------------------
df.to_parquet("appliances_with_sentiment.parquet", index=False)
print("✅ Saved: appliances_with_sentiment.parquet")
