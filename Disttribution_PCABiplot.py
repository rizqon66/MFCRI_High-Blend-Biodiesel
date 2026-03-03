import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re # Added for cleaning column names
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

# =========================
# Load & Clean Data (Replicating steps from earlier cells)
# =========================
path = "00_DATABASE_UJI_ASDP2024-1.xlsx"
raw = pd.read_excel(path, sheet_name="DB_MAIN", header=None)

# header row 0, unit rows 1-2, data start row 3
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

# Define the feature columns based on actual names
features = ["TAN","Viscosity @ 40 deg C","Water Content","%FAME"]
for c in features:
    df[c] = df[c].apply(to_num)

# Ensure all features are numeric by dropping rows with NaNs in these columns for the calculations below
df = df.dropna(subset=features).copy() # Use df, not df2, since this cell reinitializes it

# -- Figure 7: MFCRI Distribution --
# Calculate MFCRI *before* using it in correlation heatmap
df['MFCRI'] = 0.35*pd.Series(np.clip((df['Water Content']-0.2)/0.4,0,1)) \
             + 0.30*df['TAN'].rank(pct=True) \
             + 0.20*df['Viscosity @ 40 deg C'].rank(pct=True) \
             + 0.15*df['%FAME'].rank(pct=True)

# -- Table 3: Descriptive Stats --
stats = df[features].agg(['mean','std','min','max'])
# Compute %out-of-spec (assuming SNI limits: TAN<=0.50, Water<=0.10, Viscosity 2.0-4.5 cSt, FAME>=96.5)
limits = {
    'TAN':0.50,
    'Water Content':0.10,
    'Viscosity @ 40 deg C_min':2.0,
    'Viscosity @ 40 deg C_max':4.5,
    '%FAME_min':96.5
}
outspec = {
    'TAN': 100*np.mean(df['TAN']>limits['TAN']),
    'Water Content': 100*np.mean(df['Water Content']>limits['Water Content']),
    'Viscosity @ 40 deg C': 100*(np.mean(df['Viscosity @ 40 deg C']<limits['Viscosity @ 40 deg C_min']) + \
                                   np.mean(df['Viscosity @ 40 deg C']>limits['Viscosity @ 40 deg C_max'])),
    '%FAME':100*np.mean(df['%FAME']<limits['%FAME_min'])
}
table3 = pd.DataFrame({
    'Mean': stats.loc['mean'],
    'StdDev': stats.loc['std'],
    'Min': stats.loc['min'],
    'Max': stats.loc['max'],
    '%>Limit': [outspec['TAN'], outspec['Water Content'], outspec['Viscosity @ 40 deg C'], outspec['%FAME']]
})
print(table3.round(2))

# -- Figure 4: Parameter Distributions --
plt.figure(figsize=(8,6))
sns.histplot(df['TAN'], color='skyblue', label='TAN', kde=False, bins=10)
plt.axvline(0.50, color='red', linestyle='--', label='SNI Limit')
plt.xlabel('TAN (mg KOH/g)'); plt.legend()
plt.savefig('fig_TAN_dist.png')

plt.figure(figsize=(8,6))
sns.histplot(df['Water Content'], color='seagreen', label='Water', kde=False, bins=10)
plt.axvline(200, color='red', linestyle='--', label='SNI Limit')
plt.xlabel('Water Content (% vol)'); plt.legend()
plt.savefig('fig_Water_dist.png')

plt.figure(figsize=(8,6))
sns.histplot(df['Viscosity @ 40 deg C'], color='orange', label='Viscosity', kde=False, bins=10)
plt.axvline(2.0, color='red', linestyle='--', label='Min Spec')
plt.axvline(4.5, color='red', linestyle='--', label='Max Spec')
plt.xlabel('Viscosity @40°C (cSt)'); plt.legend()
plt.savefig('fig_Visc_dist.png')

plt.figure(figsize=(8,6))
sns.histplot(df['%FAME'], color='purple', label='FAME%', kde=False, bins=10)
plt.axvline(96.5, color='red', linestyle='--', label='Min Spec')
plt.xlabel('FAME Content (%)'); plt.legend()
plt.savefig('fig_FAME_dist.png')

# -- Figure 5: Correlation Heatmap --
features1 = ["TAN","Viscosity @ 40 deg C","Water Content","%FAME","MFCRI"]
corr = df[features1].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation of Fuel Parameters')
plt.savefig('fig_correlation_heatmap.png')

# -- Figure 6: PCA Biplot --
X = df[features].values
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
plt.figure(figsize=(10,8))
plt.scatter(pcs[:,0], pcs[:,1], c='blue', s=50)
for i, param in enumerate(features):
    x_offset = 0
    y_offset = 0
    # Adjust offsets for problematic labels to prevent overlap
    if param == '%FAME':
        x_offset = 0.1 # Shift right
        y_offset = 0.05 # Shift up slightly
    elif param == 'Viscosity @ 40 deg C':
        x_offset = 30.5 # Shift left
        y_offset = -0.05 # Shift down slightly
    elif param == 'Water Content':
        x_offset = 0.05 # Shift right slightly
        y_offset = -0.2 # Shift down
    elif param == 'TAN':
        x_offset = -38.05 # Shift right slightly
        y_offset = -0.05 # Shift down

    plt.arrow(0, 0, pca.components_[0,i]*2, pca.components_[1,i]*2,
              color='red', width=0.01, head_width=0.1)
    plt.text(pca.components_[0,i]*2.2 + x_offset, pca.components_[1,i]*2.2 + y_offset, param, color='red')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('PCA Biplot of Fuel Properties')
plt.savefig('fig_PCA_biplot.png')

plt.figure(figsize=(6,4))
sns.histplot(df['MFCRI'], color='gray', bins=10)
plt.axvline(df['MFCRI'].quantile(0.33), color='red', linestyle='--', label='Low/Med cutoff')
plt.axvline(df['MFCRI'].quantile(0.66), color='red', linestyle='--', label='Med/High cutoff')
plt.xlabel('MFCRI (Risk Index)'); plt.legend()
plt.savefig('fig_MFCRI_histogram.png')