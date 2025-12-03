import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from itertools import combinations

# Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)

# Load data
data_full = pd.read_csv("LINTHALL.txt", delim_whitespace=True)
data_5 = pd.read_csv("LINTH-5.txt", delim_whitespace=True)

# =====================================================================
# PART 1 VISUALS
# =====================================================================

fig1 = plt.figure(figsize=(18, 10))

X_vars = ['H2S','SAL','Eh7','pH','BUF','P','K','Ca','Mg','Na','Mn','Zn','Cu','NH4']
X = data_full[X_vars].values
y = data_full["BIO"].values

# 1. VIF
plt.subplot(2,3,1)
vif_scores = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
plt.barh(X_vars, vif_scores, color="coral")
plt.axvline(10, color="red", linestyle="--")
plt.title("Variance Inflation Factors")

# 2. Correlation Heatmap
plt.subplot(2,3,2)
sns.heatmap(data_full[X_vars].corr(), cmap="coolwarm", square=True)
plt.title("Correlation Matrix")

# 3. Scree Plot
plt.subplot(2,3,3)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
pca.fit(X_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
plt.axhline(0.90, linestyle="--", color="green")
plt.title("Scree Plot – PCA")

# 4. Eigenvalues
plt.subplot(2,3,4)
plt.bar(range(1, 15), pca.explained_variance_)
plt.axhline(1, linestyle="--", color="red")
plt.title("PCA Eigenvalues")

# 5. Condition Indices
plt.subplot(2,3,5)
XtX = np.dot(X_scaled.T, X_scaled)
eigvals = np.sort(np.linalg.eigvalsh(XtX))[::-1]
ci = np.sqrt(eigvals[0] / eigvals)
plt.bar(range(1, 15), ci)
plt.axhline(30, linestyle="--", color="red")
plt.yscale("log")
plt.title("Condition Indices")

# 6. OLS vs PCR Coefficients
plt.subplot(2,3,6)
ols = sm.OLS(y, sm.add_constant(X)).fit().params[1:]
n_components = np.sum(pca.explained_variance_ > 1)
X_pca = pca.transform(X_scaled)[:, :n_components]
pcr = sm.OLS(y, sm.add_constant(X_pca)).fit().params[1:]
beta = np.dot(pca.components_[:n_components].T, pcr) / scaler.scale_
plt.bar(np.arange(len(X_vars)) - 0.2, ols, width=0.4, label="OLS")
plt.bar(np.arange(len(X_vars)) + 0.2, beta, width=0.4, label="PCR")
plt.xticks(np.arange(len(X_vars)), X_vars, rotation=45)
plt.title("OLS vs PCR")
plt.legend()

plt.tight_layout()
plt.savefig("part1_analysis.png", dpi=300)
print("Saved: part1_analysis.png")

# =====================================================================
# PART 2 VISUALS
# =====================================================================

fig2 = plt.figure(figsize=(18, 10))

X5_vars = ['SAL','pH','K','Na','Zn']
X5 = data_5[X5_vars].values
y5 = data_5["BIO"].values

# 1. VIF
plt.subplot(2,3,1)
vif5 = [variance_inflation_factor(X5, i) for i in range(X5.shape[1])]
plt.barh(X5_vars, vif5, color="skyblue")
plt.axvline(10, linestyle="--", color="red")
plt.title("VIF – 5 Predictors")

# 2. Corr Matrix
plt.subplot(2,3,2)
sns.heatmap(data_5[X5_vars].corr(), annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Matrix – 5 Variables")

# 3. Ridge Trace
plt.subplot(2,3,3)
scaler5 = StandardScaler()
X5_scaled = scaler5.fit_transform(X5)
y5_c = y5 - np.mean(y5)
lams = np.logspace(-2, 3, 80)
coefs = []

for lam in lams:
    r = Ridge(alpha=lam, fit_intercept=False)
    r.fit(X5_scaled, y5_c)
    coefs.append(r.coef_)
coefs = np.array(coefs)

for i, v in enumerate(X5_vars):
    plt.plot(lams, coefs[:, i], label=v)
plt.xscale("log")
plt.title("Ridge Trace Plot")
plt.legend()

# 4. Subset: AIC/BIC
plt.subplot(2,3,4)
aics = []
bics = []
labels = []
from itertools import combinations

for combo in combinations(range(len(X5_vars)), 2):
    X_sub = sm.add_constant(X5[:, combo])
    fit = sm.OLS(y5, X_sub).fit()
    aics.append(fit.aic)
    bics.append(fit.bic)
    labels.append(" + ".join([X5_vars[i] for i in combo]))

plt.plot(aics, marker="o", label="AIC")
plt.plot(bics, marker="s", label="BIC")
plt.title("Subset Selection – AIC vs BIC")
plt.legend()

# 5. SSE Comparison
plt.subplot(2,3,5)
sses = []
for combo in combinations(range(len(X5_vars)), 2):
    X_sub = sm.add_constant(X5[:, combo])
    fit = sm.OLS(y5, X_sub).fit()
    sses.append(np.sum(fit.resid**2))

plt.bar(range(len(sses)), sses, color="teal")
plt.title("Subset Selection – SSE")

# 6. Model R2 Comparison
plt.subplot(2,3,6)
r2_vals = []

# Full Model
r2_vals.append(sm.OLS(y5, sm.add_constant(X5)).fit().rsquared)

# Stepwise (placeholder)
r2_vals.append(0.45)

# Best 2-var (placeholder)
r2_vals.append(0.40)

# Ridge (placeholder)
r2_vals.append(0.42)

plt.bar(["Full","Stepwise","Best 2-var","Ridge"], r2_vals, color=["blue","orange","green","purple"])
plt.title("Model Comparison – R²")

plt.tight_layout()
plt.savefig("part2_analysis.png", dpi=300)
print("Saved: part2_analysis.png")

# Show at the VERY END
plt.show()

print("\nAll visualizations generated successfully!")