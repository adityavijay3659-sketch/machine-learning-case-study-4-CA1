import time
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(42)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# 1800 Amazon reviews: 1-2 stars = negative, 3 = neutral, 4-5 = positive
df = pd.read_csv("dataset.csv")
print(df.head())
print(df["sentiment"].value_counts())
labels = ["negative", "neutral", "positive"]
df["label"] = df["sentiment"].map({s: i for i, s in enumerate(labels)})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"].tolist(), df["label"].values, test_size=0.2, stratify=df["label"], random_state=42)


def batches(n, size=32):
    idx = np.random.permutation(n)
    return [idx[i:i + size] for i in range(0, n, size)]


# ---------------- LSTM baseline (trained from scratch) ----------------
def tokens(text):
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").split()

counts = Counter(w for t in X_train for w in tokens(t))
words = [w for w, c in counts.items() if c >= 2]
vocab = {w: i + 2 for i, w in enumerate(words)}   # 0 = padding, 1 = unknown word

def encode(texts, max_len=60):
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in tokens(t)][:max_len]
        out[i, :len(ids)] = ids
    return torch.tensor(out)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 100, padding_idx=0)
        self.lstm = nn.LSTM(100, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.fc(torch.cat([h[0], h[1]], dim=1))

Xtr, Xte = encode(X_train), encode(X_test)
ytr = torch.tensor(y_train)
lstm = LSTMClassifier(len(vocab) + 2).to(device)
opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(8):
    lstm.train()
    for b in batches(len(Xtr)):
        opt.zero_grad()
        loss = loss_fn(lstm(Xtr[b].to(device)), ytr[b].to(device))
        loss.backward()
        opt.step()
    print("LSTM epoch", epoch + 1, "loss", round(loss.item(), 3))
lstm_time = time.time() - start

lstm.eval()
with torch.no_grad():
    lstm_pred = lstm(Xte.to(device)).argmax(1).cpu().numpy()

# ---------------- BERT fine-tuning (transfer learning) ----------------
name = "google/bert_uncased_L-4_H-256_A-4"   # small pre-trained BERT, runs fine on a laptop
tok = AutoTokenizer.from_pretrained(name)
bert = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3).to(device)
opt = torch.optim.AdamW(bert.parameters(), lr=5e-5)

enc_train = tok(X_train, padding=True, truncation=True, max_length=96, return_tensors="pt")
enc_test = tok(X_test, padding=True, truncation=True, max_length=96, return_tensors="pt")

start = time.time()
for epoch in range(3):
    bert.train()
    for b in batches(len(X_train), 16):
        opt.zero_grad()
        out = bert(input_ids=enc_train["input_ids"][b].to(device),
                   attention_mask=enc_train["attention_mask"][b].to(device),
                   labels=ytr[b].to(device))
        out.loss.backward()
        opt.step()
    print("BERT epoch", epoch + 1, "loss", round(out.loss.item(), 3))
bert_time = time.time() - start

bert.eval()
with torch.no_grad():
    logits = bert(input_ids=enc_test["input_ids"].to(device),
                  attention_mask=enc_test["attention_mask"].to(device)).logits
bert_pred = logits.argmax(1).cpu().numpy()

# ---------------- comparison ----------------
for model_name, pred, t, m in [("LSTM", lstm_pred, lstm_time, lstm), ("BERT", bert_pred, bert_time, bert)]:
    print("\n=====", model_name, "=====")
    print("accuracy:", round(accuracy_score(y_test, pred), 3))
    print("train time (s):", round(t, 1), " parameters:", sum(p.numel() for p in m.parameters()))
    print(classification_report(y_test, pred, target_names=labels))
    print(confusion_matrix(y_test, pred))

print("\nBERT already learned English from millions of sentences, so with only 1440 training reviews")
print("and 3 epochs it beats an LSTM that has to learn word meanings from scratch (transfer learning).")
print("Neutral is the hardest class for both models because 3-star reviews mix praise and complaints.")
print("Deployment: bert-base is ~110M params and slow on CPU, so in production use a small/distilled BERT,")
print("quantize it, cap max_length, batch requests, or use GPU. LSTM is tiny and fast but less accurate.")
print("Also watch for data drift (new slang, new products) and retrain periodically.")
