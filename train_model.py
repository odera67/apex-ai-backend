import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
import os

print("🚀 Starting Training Pipeline...")

# ==========================================
# 1. WORKOUT MODEL
# ==========================================
print("🏋️ Training Workout Model...")
# Ensure it loads your realistic dataset
df_workouts = pd.read_csv("1200_Realistic_Workouts_Dataset.csv").fillna("")

mlb_goal = MultiLabelBinarizer()
mlb_diff = MultiLabelBinarizer()
mlb_focus = MultiLabelBinarizer()
mlb_equip = MultiLabelBinarizer()

X_goal = mlb_goal.fit_transform(df_workouts['Goal'].apply(lambda x: [x]))
X_diff = mlb_diff.fit_transform(df_workouts['Difficulty'].apply(lambda x: [x]))
X_focus = mlb_focus.fit_transform(df_workouts['Focus_Area'].apply(lambda x: [x]))

if 'Equipment' not in df_workouts.columns:
    df_workouts['Equipment'] = "bodyweight"

X_equip = mlb_equip.fit_transform(df_workouts['Equipment'].apply(lambda x: [x]))
X_workouts = np.hstack((X_goal, X_diff, X_focus, X_equip))

# 🚀 Increased n_neighbors to 20 so we have a large pool to shuffle!
model_workouts = NearestNeighbors(n_neighbors=20, metric='cosine')
model_workouts.fit(X_workouts)

# ==========================================
# 2. DIET MODEL
# ==========================================
print("🍎 Training Diet Model...")
# Ensure it loads your African Meals dataset
df_diet = pd.read_csv("1250_African_Meals_Dataset.csv").fillna("")

scaler_diet = StandardScaler()
X_diet = scaler_diet.fit_transform(df_diet[['Protein(g)', 'Carbs(g)', 'Fat(g)', 'Calories']])

# 🚀 Increased n_neighbors to 20 for diet randomization
model_diet = NearestNeighbors(n_neighbors=20, metric='euclidean')
model_diet.fit(X_diet)

# ==========================================
# 3. NLP DATASETS & DICTIONARY
# ==========================================
print("🏥 Setting up Injury, Allergy, and Dictionary Datasets...")

if os.path.exists("1250_Injuries_Modifications_Dataset.csv"):
    joblib.dump(pd.read_csv("1250_Injuries_Modifications_Dataset.csv").fillna(""), "df_injuries.pkl")
else:
    joblib.dump(pd.DataFrame(), "df_injuries.pkl")

if os.path.exists("1250_Allergies_Dietary_Restrictions_Dataset.csv"):
    joblib.dump(pd.read_csv("1250_Allergies_Dietary_Restrictions_Dataset.csv").fillna(""), "df_allergies.pkl")
else:
    joblib.dump(pd.DataFrame(), "df_allergies.pkl")

# 🚀 NEW: Load the Food Dictionary
if os.path.exists("FoodData.csv"):
    joblib.dump(pd.read_csv("FoodData.csv").fillna(""), "df_food_dictionary.pkl")
else:
    joblib.dump(pd.DataFrame(), "df_food_dictionary.pkl")

# ==========================================
# 4. SAVE EVERYTHING
# ==========================================
print("💾 Saving Models...")

joblib.dump(model_workouts, "model_workouts.pkl")
joblib.dump(mlb_goal, "mlb_goal.pkl")
joblib.dump(mlb_diff, "mlb_diff.pkl")
joblib.dump(mlb_focus, "mlb_focus.pkl")
joblib.dump(mlb_equip, "mlb_equip.pkl")

joblib.dump(model_diet, "model_diet.pkl")
joblib.dump(scaler_diet, "scaler_diet.pkl")

df_workouts.to_pickle("df_workouts_reference.pkl")
df_diet.to_pickle("df_diet_reference.pkl")

print("✅ Training Pipeline Complete!")