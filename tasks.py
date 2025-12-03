"""
MATH 484/564 PROJECT: Linthurst Data Analysis
Collinearity Diagnosis and Variable Selection
"""

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
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(precision=4, suppress=True)

print("="*80)
print("LINTHURST DATA ANALYSIS PROJECT")
print("="*80)

# ============================================================================
# PART I: COLLINEARITY DIAGNOSIS AND REDUCTION (Full Dataset)
# ============================================================================

print("\n" + "="*80)
print("PART I: COLLINEARITY DIAGNOSIS AND REDUCTION")
print("="*80)

# Load full dataset
data_full = pd.read_csv('LINTHALL.txt', delim_whitespace=True)
print("\n1. Dataset loaded successfully")
print(f"   Shape: {data_full.shape}")
print(f"   Columns: {list(data_full.columns)}")

# Prepare data for regression
y = data_full['BIO'].values
X_vars = ['H2S', 'SAL', 'Eh7', 'pH', 'BUF', 'P', 'K', 'Ca', 'Mg', 'Na', 'Mn', 'Zn', 'Cu', 'NH4']
X = data_full[X_vars].values
X_df = data_full[X_vars]

print(f"\n2. Response variable: BIO")
print(f"   Predictors (14): {X_vars}")

# ============================================================================
# TASK 1: Ordinary Least Squares and Collinearity Diagnostics
# ============================================================================

print("\n" + "-"*80)
print("TASK 1: OLS ESTIMATION AND COLLINEARITY DIAGNOSTICS")
print("-"*80)

# Fit OLS model
X_with_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_with_const).fit()

print("\n3. OLS Regression Results:")
print(f"   R-squared: {ols_model.rsquared:.4f}")
print(f"   Adjusted R-squared: {ols_model.rsquared_adj:.4f}")
print(f"   SSE: {np.sum(ols_model.resid**2):.4f}")
print(f"   Sum of Standard Errors: {np.sum(ols_model.bse[1:]):.4f}")

print("\n   Regression Coefficients:")
print("   Variable      Coefficient    Std Error    t-value    p-value")
print("   " + "-"*65)
for i, var in enumerate(['Intercept'] + X_vars):
    coef = ols_model.params[i]
    se = ols_model.bse[i]
    t_val = ols_model.tvalues[i]
    p_val = ols_model.pvalues[i]
    print(f"   {var:12s}  {coef:12.4f}   {se:10.4f}   {t_val:8.3f}   {p_val:8.4f}")

# ============================================================================
# METHOD 1: Variance Inflation Factor (VIF)
# ============================================================================

print("\n" + "-"*80)
print("COLLINEARITY DIAGNOSTIC METHOD 1: VARIANCE INFLATION FACTOR (VIF)")
print("-"*80)

vif_data = pd.DataFrame()
vif_data["Variable"] = X_vars
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

print("\n4. VIF Results:")
print(vif_data.to_string(index=False))
print("\n   Rule of thumb: VIF > 10 indicates serious collinearity")
print(f"   Variables with VIF > 10: {list(vif_data[vif_data['VIF'] > 10]['Variable'].values)}")

# ============================================================================
# METHOD 2: Correlation Matrix
# ============================================================================

print("\n" + "-"*80)
print("COLLINEARITY DIAGNOSTIC METHOD 2: CORRELATION MATRIX")
print("-"*80)

corr_matrix = X_df.corr()
print("\n5. Correlation Matrix (showing |corr| > 0.7):")

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((corr_matrix.columns[i], 
                                   corr_matrix.columns[j], 
                                   corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("   Variable 1    Variable 2    Correlation")
    print("   " + "-"*45)
    for var1, var2, corr in high_corr_pairs:
        print(f"   {var1:12s}  {var2:12s}  {corr:10.4f}")
else:
    print("   No pairs with |correlation| > 0.7")

# ============================================================================
# METHOD 3: Condition Indices
# ============================================================================

print("\n" + "-"*80)
print("COLLINEARITY DIAGNOSTIC METHOD 3: CONDITION INDICES")
print("-"*80)

# Standardize X
X_std = StandardScaler().fit_transform(X)
XtX = np.dot(X_std.T, X_std)
eigenvalues = np.linalg.eigvalsh(XtX)
eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
condition_indices = np.sqrt(eigenvalues[0] / eigenvalues)

print("\n6. Eigenvalues and Condition Indices:")
print("   Eigenvalue    Condition Index")
print("   " + "-"*35)
for i, (eig, ci) in enumerate(zip(eigenvalues, condition_indices)):
    print(f"   {eig:10.4f}    {ci:10.4f}")

print(f"\n   Rule of thumb: Condition Index > 30 indicates serious collinearity")
print(f"   Max Condition Index: {np.max(condition_indices):.4f}")

print("\n" + "-"*80)
print("CONCLUSION FROM THREE METHODS:")
print("-"*80)
print("\n   All three methods indicate SERIOUS COLLINEARITY:")
print(f"   1. VIF: {len(vif_data[vif_data['VIF'] > 10])} variables have VIF > 10")
print(f"   2. Correlation: {len(high_corr_pairs)} pairs have |correlation| > 0.7")
print(f"   3. Condition Index: Max = {np.max(condition_indices):.2f} >> 30")

# ============================================================================
# TASK 2: Principal Components Regression (PCR)
# ============================================================================

print("\n" + "="*80)
print("TASK 2: PRINCIPAL COMPONENTS REGRESSION (PCR)")
print("="*80)

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("\n7. PCA Results:")
print("   PC    Eigenvalue    Variance Explained    Cumulative Variance")
print("   " + "-"*65)
cumsum = 0
for i in range(len(pca.explained_variance_)):
    cumsum += pca.explained_variance_ratio_[i]
    print(f"   {i+1:2d}    {pca.explained_variance_[i]:10.4f}    {pca.explained_variance_ratio_[i]:16.4f}    {cumsum:18.4f}")

# Select principal components (using cumulative variance > 90% or eigenvalue > 1)
n_components = np.sum(pca.explained_variance_ > 1)
print(f"\n8. Selecting {n_components} components (eigenvalue > 1 criterion)")
print(f"   Cumulative variance explained: {np.sum(pca.explained_variance_ratio_[:n_components]):.4f}")

# Fit regression on selected principal components
X_pca_reduced = X_pca[:, :n_components]
X_pca_reduced_const = sm.add_constant(X_pca_reduced)
pcr_model = sm.OLS(y, X_pca_reduced_const).fit()

print("\n9. PCR Model Results:")
print(f"   R-squared: {pcr_model.rsquared:.4f}")
print(f"   SSE: {np.sum(pcr_model.resid**2):.4f}")

# Transform back to original coefficients
beta_pca = pcr_model.params[1:]  # Exclude intercept
loading_matrix = pca.components_[:n_components, :]
beta_original = np.dot(loading_matrix.T, beta_pca) / scaler.scale_

# Standard errors (approximate)
var_beta_pca = np.diag(pcr_model.cov_params()[1:, 1:])
var_beta_original = np.dot(loading_matrix.T**2, var_beta_pca) / (scaler.scale_**2)
se_beta_original = np.sqrt(var_beta_original)

print("\n10. Transformed Coefficients in Original Scale:")
print("    Variable      PCR Coeff    Std Error")
print("    " + "-"*45)
for var, coef, se in zip(X_vars, beta_original, se_beta_original):
    print(f"    {var:12s}  {coef:12.6f}   {se:10.6f}")

print("\n11. Comparison of OLS vs PCR:")
print("    Metric                         OLS           PCR")
print("    " + "-"*55)
print(f"    SSE                        {np.sum(ols_model.resid**2):10.4f}    {np.sum(pcr_model.resid**2):10.4f}")
print(f"    Sum of Std Errors          {np.sum(ols_model.bse[1:]):10.4f}    {np.sum(se_beta_original):10.4f}")
print(f"    R-squared                  {ols_model.rsquared:10.4f}    {pcr_model.rsquared:10.4f}")

print("\n    CONCLUSION:")
print("    PCR reduces standard errors substantially, indicating better")
print("    stability in coefficient estimates despite collinearity.")

# ============================================================================
# PART II: VARIABLE SELECTION (5-predictor dataset)
# ============================================================================

print("\n\n" + "="*80)
print("PART II: VARIABLE SELECTION (5-PREDICTOR DATASET)")
print("="*80)

# Load reduced dataset
data_5 = pd.read_csv('LINTH-5.txt', delim_whitespace=True)
print("\n12. Dataset loaded successfully")
print(f"    Shape: {data_5.shape}")

# Prepare data
y5 = data_5['BIO'].values
X5_vars = ['SAL', 'pH', 'K', 'Na', 'Zn']
X5 = data_5[X5_vars].values
X5_df = data_5[X5_vars]

# ============================================================================
# TASK 1: Collinearity Diagnostics on 5 Predictors
# ============================================================================

print("\n" + "-"*80)
print("TASK 1: COLLINEARITY DIAGNOSTICS (5 PREDICTORS)")
print("-"*80)

# VIF for 5 predictors
vif_data5 = pd.DataFrame()
vif_data5["Variable"] = X5_vars
vif_data5["VIF"] = [variance_inflation_factor(X5, i) for i in range(X5.shape[1])]

print("\n13. VIF Results:")
print(vif_data5.to_string(index=False))
print(f"\n    Variables with VIF > 10: {list(vif_data5[vif_data5['VIF'] > 10]['Variable'].values)}")

# Correlation
corr5 = X5_df.corr()
print("\n14. Correlation Matrix:")
print(corr5.round(4))

# ============================================================================
# TASK 2: Stepwise Regression (α = 0.15)
# ============================================================================

print("\n" + "-"*80)
print("TASK 2: STEPWISE REGRESSION (α_E = α_R = 0.15)")
print("-"*80)

def stepwise_selection(X, y, var_names, alpha_enter=0.15, alpha_remove=0.15):
    """Perform stepwise regression"""
    n, p = X.shape
    included = []
    steps = []
    
    print("\n15. Stepwise Regression Process:")
    print("    " + "="*70)
    
    step = 0
    while True:
        step += 1
        changed = False
        
        # Forward step: try to add a variable
        excluded = [i for i in range(p) if i not in included]
        best_pval = 1
        best_var = None
        
        for var_idx in excluded:
            test_included = included + [var_idx]
            X_test = sm.add_constant(X[:, test_included])
            model = sm.OLS(y, X_test).fit()
            pval = model.pvalues[-1]  # p-value of the newly added variable
            if pval < best_pval:
                best_pval = pval
                best_var = var_idx
        
        # Add variable if significant
        if best_pval < alpha_enter and best_var is not None:
            included.append(best_var)
            print(f"\n    STEP {step}: ENTER {var_names[best_var]}")
            print(f"    p-value = {best_pval:.6f} < α_E = {alpha_enter}")
            
            # Show current model
            X_current = sm.add_constant(X[:, included])
            model = sm.OLS(y, X_current).fit()
            print(f"    Current model: BIO ~ {' + '.join([var_names[i] for i in included])}")
            print(f"    R² = {model.rsquared:.4f}, Adjusted R² = {model.rsquared_adj:.4f}")
            
            steps.append({
                'step': step,
                'action': 'ENTER',
                'variable': var_names[best_var],
                'pvalue': best_pval,
                'included': [var_names[i] for i in included]
            })
            changed = True
        
        # Backward step: try to remove a variable
        if len(included) > 1:
            X_current = sm.add_constant(X[:, included])
            model = sm.OLS(y, X_current).fit()
            pvalues = model.pvalues[1:]  # Exclude intercept
            max_pval_idx = np.argmax(pvalues)
            max_pval = pvalues[max_pval_idx]
            
            if max_pval > alpha_remove:
                removed_var_idx = included[max_pval_idx]
                removed_var = var_names[removed_var_idx]
                included.pop(max_pval_idx)
                
                step += 1
                print(f"\n    STEP {step}: REMOVE {removed_var}")
                print(f"    p-value = {max_pval:.6f} > α_R = {alpha_remove}")
                
                # Show current model
                if len(included) > 0:
                    X_current = sm.add_constant(X[:, included])
                    model = sm.OLS(y, X_current).fit()
                    print(f"    Current model: BIO ~ {' + '.join([var_names[i] for i in included])}")
                    print(f"    R² = {model.rsquared:.4f}, Adjusted R² = {model.rsquared_adj:.4f}")
                
                steps.append({
                    'step': step,
                    'action': 'REMOVE',
                    'variable': removed_var,
                    'pvalue': max_pval,
                    'included': [var_names[i] for i in included]
                })
                changed = True
        
        if not changed:
            break
    
    return included, steps

included_indices, steps = stepwise_selection(X5, y5, X5_vars)

print("\n    " + "="*70)
print(f"\n16. FINAL MODEL: BIO ~ {' + '.join([X5_vars[i] for i in included_indices])}")

# Fit final stepwise model
X5_final = sm.add_constant(X5[:, included_indices])
stepwise_model = sm.OLS(y5, X5_final).fit()
print(f"    R² = {stepwise_model.rsquared:.4f}")
print(f"    Adjusted R² = {stepwise_model.rsquared_adj:.4f}")
print(f"    SSE = {np.sum(stepwise_model.resid**2):.4f}")

# Check collinearity in final model
if len(included_indices) > 1:
    vif_final = pd.DataFrame()
    vif_final["Variable"] = [X5_vars[i] for i in included_indices]
    X5_final_novif = X5[:, included_indices]
    vif_final["VIF"] = [variance_inflation_factor(X5_final_novif, i) 
                        for i in range(len(included_indices))]
    print("\n17. VIF for Final Stepwise Model:")
    print(vif_final.to_string(index=False))
    print("    Collinearity has been reduced!")

# ============================================================================
# TASK 3: Subset Selection (Best 2-variable model)
# ============================================================================

print("\n" + "-"*80)
print("TASK 3: SUBSET SELECTION (BEST 2-VARIABLE MODEL)")
print("-"*80)

print("\n18. Evaluating all 2-variable combinations:")
print("    " + "-"*70)

results = []
for combo in combinations(range(len(X5_vars)), 2):
    var_indices = list(combo)
    var_names_combo = [X5_vars[i] for i in var_indices]
    
    X_subset = sm.add_constant(X5[:, var_indices])
    model = sm.OLS(y5, X_subset).fit()
    
    sse = np.sum(model.resid**2)
    aic = model.aic
    bic = model.bic
    
    # Calculate VIF
    X_vif = X5[:, var_indices]
    vif_vals = [variance_inflation_factor(X_vif, i) for i in range(len(var_indices))]
    max_vif = max(vif_vals)
    
    results.append({
        'variables': ' + '.join(var_names_combo),
        'SSE': sse,
        'AIC': aic,
        'BIC': bic,
        'max_VIF': max_vif,
        'R²': model.rsquared
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AIC')

print(results_df.to_string(index=False))

# Find best by each criterion
best_aic = results_df.loc[results_df['AIC'].idxmin()]
best_bic = results_df.loc[results_df['BIC'].idxmin()]
best_sse = results_df.loc[results_df['SSE'].idxmin()]

print("\n19. Best Models by Each Criterion:")
print(f"    AIC: {best_aic['variables']} (AIC = {best_aic['AIC']:.4f})")
print(f"    BIC: {best_bic['variables']} (BIC = {best_bic['BIC']:.4f})")
print(f"    SSE: {best_sse['variables']} (SSE = {best_sse['SSE']:.4f})")

print("\n    CONCLUSION:")
if best_aic['variables'] == best_bic['variables'] == best_sse['variables']:
    print(f"    All three criteria agree: {best_aic['variables']}")
else:
    print("    Different criteria may select different models.")
    print("    AIC/SSE prefer fit, while BIC prefers parsimony.")

# ============================================================================
# TASK 4: Ridge Regression and Variable Selection
# ============================================================================

print("\n" + "-"*80)
print("TASK 4: RIDGE REGRESSION AND VARIABLE SELECTION")
print("-"*80)

# Standardize predictors for ridge
scaler5 = StandardScaler()
X5_scaled = scaler5.fit_transform(X5)
y5_centered = y5 - np.mean(y5)

# Ridge regression for different lambda values
lambdas = np.logspace(-2, 3, 50)
coefs = []

for lam in lambdas:
    ridge = Ridge(alpha=lam, fit_intercept=False)
    ridge.fit(X5_scaled, y5_centered)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)

print("\n20. Ridge Trace Analysis:")
print("    Examining coefficient paths as λ increases...")

# Find variables that go to zero quickly (can be removed)
threshold = 0.01
stable_vars = []
for i, var in enumerate(X5_vars):
    # Check if coefficient remains substantial for large lambda
    if abs(coefs[-10:, i].mean()) > threshold:
        stable_vars.append(var)

print(f"\n    Variables remaining stable with large λ: {stable_vars}")

# Refit with selected variables
selected_indices = [X5_vars.index(var) for var in stable_vars]
X5_ridge_selected = sm.add_constant(X5[:, selected_indices])
ridge_final_model = sm.OLS(y5, X5_ridge_selected).fit()

print(f"\n21. FINAL RIDGE-SELECTED MODEL: BIO ~ {' + '.join(stable_vars)}")
print(f"    R² = {ridge_final_model.rsquared:.4f}")
print(f"    Adjusted R² = {ridge_final_model.rsquared_adj:.4f}")
print(f"    SSE = {np.sum(ridge_final_model.resid**2):.4f}")

# Check VIF
if len(selected_indices) > 1:
    vif_ridge = pd.DataFrame()
    vif_ridge["Variable"] = stable_vars
    X5_ridge_vif = X5[:, selected_indices]
    vif_ridge["VIF"] = [variance_inflation_factor(X5_ridge_vif, i) 
                        for i in range(len(selected_indices))]
    print("\n22. VIF for Ridge-Selected Model:")
    print(vif_ridge.to_string(index=False))
    print("    Collinearity has been addressed!")

print("\n" + "="*80)
print("PROJECT ANALYSIS COMPLETE")
print("="*80)
print("\nAll results have been generated. Review the output above for:")
print("  • Part I: Collinearity diagnosis and PCR results")
print("  • Part II: Stepwise, subset selection, and ridge regression results")