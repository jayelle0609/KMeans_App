import pandas as pd
import numpy as np

np.random.seed(42)

n_rows = 100

# -------------------------
# Base data
# -------------------------
age = np.random.randint(18, 60, size=n_rows)
gender = np.random.choice(['Male', 'Female'], size=n_rows)
city = np.random.choice(['A', 'B', 'C'], size=n_rows, p=[0.4, 0.35, 0.25])
education = np.random.choice(
    ['Secondary School', 'ITE', 'Diploma', 'Bachelor', 'Master'],
    size=n_rows,
    p=[0.4, 0.25, 0.2, 0.1, 0.05]   # adjusted so probabilities sum to 1
)

# -------------------------
# Encode categorical values numerically for pattern generation
# -------------------------
city_score = pd.Series(city).map({'A': 3, 'B': 2, 'C': 1}).values
edu_score = pd.Series(education).map({
    'Secondary School': 1,
    'ITE': 2,
    'Diploma': 3,
    'Bachelor': 4,
    'Master': 5
}).values

# -------------------------
# Generate dependent variables
# -------------------------
# Annual income increases with age, education, city quality
annual_income = 20000 + age * 500 + edu_score * 5000 + city_score * 3000 + np.random.normal(0, 2000, n_rows)

# Spending score increases with age, education, and better city
spending_score = 20 + age * 1.5 + edu_score * 5 + city_score * 3 + np.random.normal(0, 5, n_rows)

# -------------------------
# Assemble into a dataframe
# -------------------------
train_df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'city': city,
    'education': education,
    'annual_income': np.round(annual_income, 0),
    'spending_score': np.round(spending_score, 1)
})

# -------------------------
# Save to CSV
# -------------------------
train_df.to_csv('train.csv', index=False)

# Create test set (same structure but without target)
test_df = train_df.drop(columns=['spending_score'])
test_df.to_csv('test.csv', index=False)

print("✅ Sample train.csv and test.csv generated with 100 rows each.")
