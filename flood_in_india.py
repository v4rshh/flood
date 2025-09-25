import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = pd.read_csv("flood.csv")

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Drop rows with missing values
data = data.dropna()

# Normalize rainfall data
data["Rainfall"] = (data["Rainfall"] - data["Rainfall"].mean()) / data["Rainfall"].std()

# Display first 5 rows
print(data.head())

# Prior probability of flood
prior_flood = data["Floods"].mean()
print(f"Prior Probability of Flood: {prior_flood:.2f}")

def conditional_probability(feature, threshold, flood_status):
    """Compute P(feature > threshold | flood_status)."""
    subset = data[data["Floods"] == flood_status]
    return subset[subset[feature] > threshold].shape[0] / subset.shape[0]

# Example: P(Rainfall > 1.5 | Flood = 1)
rainfall_threshold = 1.5
p_rain_given_flood = conditional_probability("Rainfall", rainfall_threshold, 1)
print(f"P(Rainfall > {rainfall_threshold} | Flood = 1): {p_rain_given_flood:.2f}")

def bayesian_update(prior, likelihood, evidence):
    """Update prior probability using Bayes' theorem."""
    return (likelihood * prior) / evidence

# Compute evidence: P(Rainfall > 1.5)
p_rain_given_no_flood = conditional_probability("Rainfall", rainfall_threshold, 0)
evidence = p_rain_given_flood * prior_flood + p_rain_given_no_flood * (1 - prior_flood)

# Update flood probability
posterior_flood = bayesian_update(prior_flood, p_rain_given_flood, evidence)
print(f"Posterior Probability of Flood (Rainfall > {rainfall_threshold}): {posterior_flood:.2f}")

# Generate random dates
start_date = pd.to_datetime("1990-01-01")
end_date = pd.to_datetime("2025-01-01")
data["Date"] = pd.to_datetime(
    np.random.randint(start_date.value // 10**9, end_date.value // 10**9, data.shape[0]), unit="s"
)

# Sort data by date
data = data.sort_values(by="Date")

# Simulate flood risk over time
data["Flood_Risk"] = data["Rainfall"].apply(lambda x: bayesian_update(
    prior_flood,
    conditional_probability("Rainfall", x, 1),
    conditional_probability("Rainfall", x, 1) * prior_flood + conditional_probability("Rainfall", x, 0) * (1 - prior_flood)
))

# Plot flood risk over time
plt.figure(figsize=(10, 6))
plt.plot(data["Date"], data["Flood_Risk"], label="Flood Risk", color="red")
plt.scatter(data["Date"], data["Floods"], label="Actual Flood", color="blue", alpha=0.5)
plt.xlabel("Date")
plt.ylabel("Flood Risk / Occurrence")
plt.title("Flood Risk Prediction in India")
plt.legend()
plt.show()