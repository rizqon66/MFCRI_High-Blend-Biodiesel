import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV

# =========================
# Load & Clean Data
# =========================

path = "00_DATABASE_UJI_ASDP2024-1.xlsx"
raw = pd.read_excel(path, sheet_name="DB_MAIN", header=None)

cols = [str(x).strip().replace("\n"," ") if pd.notna(x) else None for x in raw.iloc[0].tolist()]
cols = [re.sub(r"\s+"," ",c) if c else None for c in cols]

df = raw.iloc[3:].copy()
df.columns = cols
df = df[df["Sample Id"].notna()].reset_index(drop=True)

def to_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s.startswith("<"):
        try: return float(s[1:]) * 0.5
        except: return np.nan
    if s.startswith(">"):
        s = s[1:]
    s = s.replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else np.nan

features = ["TAN","Viscosity @ 40 deg C","Water Content","%FAME"]
for c in features:
    df[c] = df[c].apply(to_num)

df2 = df.dropna(subset=features).copy()

# =========================
# MFCRI Construction
# =========================

def rank_norm(s):
    return s.rank(pct=True)

W = np.clip((df2["Water Content"]-200)/400,0,1)
A = rank_norm(df2["TAN"])
V = rank_norm(df2["Viscosity @ 40 deg C"])
F = rank_norm(df2["%FAME"])

df2["MFCRI"] = 0.35*W + 0.30*A + 0.20*V + 0.15*F

# =========================
# Figure 5 - MFCRI Histogram
# =========================
plt.figure(figsize=(8,5))
sns.histplot(df2["MFCRI"], kde=True)
plt.title("Distribution of MFCRI Values")
plt.xlabel("MFCRI")
plt.ylabel("Frequency")
plt.show()

# =========================
# Figure 6 - MFCRI by Vessel
# =========================
plt.figure(figsize=(10,5))
sns.boxplot(x="SHIP Name", y="MFCRI", data=df2)
plt.xticks(rotation=45)
plt.title("MFCRI Distribution by Vessel")
plt.show()

# =========================
# Risk Classes
# =========================
p33 = df2["MFCRI"].quantile(0.333)
p66 = df2["MFCRI"].quantile(0.666)

def risk_class(x):
    if x <= p33: return 0
    if x <= p66: return 1
    return 2

df2["RiskClass"] = df2["MFCRI"].apply(risk_class)

# =========================
# RandomForest Model
# =========================

X = df2[features]
y = df2["RiskClass"]
groups = df2["SHIP Name"]

rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=500,
                                  class_weight="balanced",
                                  random_state=42))
])

clf = CalibratedClassifierCV(rf, method="sigmoid", cv=3)

cv = GroupKFold(n_splits=5)

y_pred = np.zeros(len(y))
y_prob = np.zeros((len(y),3))

for tr, te in cv.split(X,y,groups):
    clf.fit(X.iloc[tr], y.iloc[tr])
    y_pred[te] = clf.predict(X.iloc[te])
    y_prob[te] = clf.predict_proba(X.iloc[te])

# =========================
# Confusion Matrix Plot
# =========================
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Baseline Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# Probability Histogram
# =========================
p_high = y_prob[:,2]

plt.figure(figsize=(8,5))
sns.histplot(p_high, bins=10)
plt.axvline(0.6, color='red', linestyle='--')
plt.title("Distribution of P(High) with Alert Threshold 0.60")
plt.show()