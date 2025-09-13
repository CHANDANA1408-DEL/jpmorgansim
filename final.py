# -------------------------
# 1. Import Libraries
# -------------------------
import pandas as pd
import numpy as np

# -------------------------
# 2. Load Dataset
# -------------------------
df = pd.read_csv("loan.csv")
df.columns = df.columns.str.strip()  # Remove spaces from headers

# Sort by FICO score
df = df.sort_values('fico_score').reset_index(drop=True)
scores = df['fico_score'].values
defaults = df['Defaulted'].values
n = len(scores)

# -------------------------
# 3. Precompute Cumulative Sums
# -------------------------
cum_defaults = np.cumsum(defaults)
cum_total = np.arange(1, n+1)

# -------------------------
# 4. Segment Log-Likelihood Function
# -------------------------
def segment_log_likelihood(i, j):
    """
    Compute log-likelihood of defaults between index i and j (exclusive)
    """
    if i == 0:
        ki = cum_defaults[j-1]
    else:
        ki = cum_defaults[j-1] - cum_defaults[i-1]
    ni = j - i
    if ni == 0:
        return 0
    pi = ki / ni
    # Avoid log(0)
    if pi == 0 or pi == 1:
        return 0
    return ki * np.log(pi) + (ni - ki) * np.log(1 - pi)

# -------------------------
# 5. Dynamic Programming for Optimal Binning
# -------------------------
def optimal_binning(n_buckets):
    # DP table: dp[i][k] = max log-likelihood using first i borrowers with k buckets
    dp = np.full((n+1, n_buckets+1), -np.inf)
    dp[0][0] = 0  # base case

    # Keep track of split points
    split = np.zeros((n+1, n_buckets+1), dtype=int)

    for i in range(1, n+1):
        for k in range(1, n_buckets+1):
            for j in range(k-1, i):
                val = dp[j][k-1] + segment_log_likelihood(j, i)
                if val > dp[i][k]:
                    dp[i][k] = val
                    split[i][k] = j
    return dp, split

# -------------------------
# 6. Recover Bucket Boundaries
# -------------------------
def get_boundaries(split, n_buckets):
    boundaries_idx = []
    i = n
    k = n_buckets
    while k > 0:
        j = split[i][k]
        boundaries_idx.append(j)
        i = j
        k -= 1
    boundaries_idx = sorted(boundaries_idx)
    boundaries = [scores[idx] for idx in boundaries_idx]
    return boundaries

# -------------------------
# 7. Assign Ratings Function
# -------------------------
def assign_rating(score, boundaries, n_buckets):
    for i, b in enumerate(boundaries):
        if score < b:
            return i+1
    return n_buckets

# -------------------------
# 8. Apply to Dataset
# -------------------------
n_buckets = 5  # Choose number of ratings
dp, split = optimal_binning(n_buckets)
boundaries = get_boundaries(split, n_buckets)

df['rating'] = df['fico_score'].apply(lambda x: assign_rating(x, boundaries, n_buckets))
df['rating'] = n_buckets - df['rating'] + 1  # Lower rating = better credit score

# -------------------------
# 9. Output Results
# -------------------------
print("Optimal Bucket Boundaries (FICO Scores):", boundaries)
print(df[['fico_score','Defaulted','rating']].head(20))

# Optional: Save rating map to CSV
df.to_csv("fico_rating_map.csv", index=False)
