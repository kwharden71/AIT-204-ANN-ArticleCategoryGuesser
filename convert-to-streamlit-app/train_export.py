# train_export.py
import json, joblib, numpy as np, torch, torch.nn as nn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string

SEED = 42
rng = np.random.default_rng(SEED)

# 1) Data
print("Retriving Data")
data = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
X_raw, y = data.data, data.target
num_classes = len(data.target_names)
print("Finished retriving data")

X_raw = [''.join(char.lower() if char not in string.punctuation else ' ' for char in text) for text in X_raw]

# 2) Vectorizer (fit once)
vectorizer = TfidfVectorizer(
    max_features=5000, lowercase=True, stop_words='english',
    strip_accents='unicode', token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)
X = vectorizer.fit_transform(X_raw).toarray()

# 3) Split
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# 4) Model
class NewsMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NewsMLP(Xtr.shape[1], num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# 5) Torch training (minimal)
def to_t(x): return torch.tensor(x, dtype=torch.float32, device=device)
def to_y(x): return torch.tensor(x, dtype=torch.long, device=device)
for epoch in range(100):
    model.train()
    logits = model(to_t(Xtr))
    loss = crit(logits, to_y(ytr))
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# 6) Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
model.eval()
with torch.no_grad():
    y_pred_train = model(to_t(Xtr)).argmax(dim=1).cpu().numpy()
    y_pred_test = model(to_t(Xte)).argmax(dim=1).cpu().numpy()

train_acc = accuracy_score(ytr, y_pred_train)
test_acc = accuracy_score(yte, y_pred_test)
test_precision = precision_score(yte, y_pred_test, average='weighted')
test_recall = recall_score(yte, y_pred_test, average='weighted')
test_f1 = f1_score(yte, y_pred_test, average='weighted')

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1: {test_f1:.4f}")

# 7) Export artifacts
torch.save(model.state_dict(), "model_state_dict.pt")
joblib.dump(vectorizer, "vectorizer.pkl")
with open("label_names.json","w") as f: json.dump(data.target_names, f)
print("Exported: model_state_dict.pt, vectorizer.pkl, label_names.json")
