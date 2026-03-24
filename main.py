from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import pandas as pd
import numpy as np
import joblib
import re
import os
import spacy
import random

# ==========================================
# 1. INITIALIZE API
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ==========================================
# 2. LOAD DATASETS & MODELS
# ==========================================
try:
    df_workouts = joblib.load("df_workouts_reference.pkl")
    df_diet = joblib.load("df_diet_reference.pkl")
    model_workouts = joblib.load("model_workouts.pkl")
    mlb_goal = joblib.load("mlb_goal.pkl")
    mlb_diff = joblib.load("mlb_diff.pkl")
    mlb_focus = joblib.load("mlb_focus.pkl")
    mlb_equip = joblib.load("mlb_equip.pkl")
    model_diet = joblib.load("model_diet.pkl")
    scaler_diet = joblib.load("scaler_diet.pkl")
    
    df_injuries = joblib.load("df_injuries.pkl") if os.path.exists("df_injuries.pkl") else pd.DataFrame()
    df_allergies = joblib.load("df_allergies.pkl") if os.path.exists("df_allergies.pkl") else pd.DataFrame()
    df_food_dict = joblib.load("df_food_dictionary.pkl") if os.path.exists("df_food_dictionary.pkl") else pd.DataFrame()
    
    try:
        nlp_extractor = joblib.load("nlp_extractor.pkl")
    except:
        nlp_extractor = None
except Exception as e:
    print(f"⚠️ Load Error: {e}")

# ==========================================
# NLP INJURY DETECTION & DATASET SEARCH
# ==========================================
def detect_injuries(text):
    if not text or str(text).lower() in ["none", "no", "nothing", "n/a"]:
        return []

    doc = nlp(str(text).lower())
    body_parts = [
        "knee", "back", "shoulder", "neck", "wrist", "ankle", "hip", "elbow", "rib", "core", "leg", "arm",
        "tibia", "fibula", "femur", "shin", "calf", "thigh", "foot", "toe", "finger", "hand", "spine", 
        "pelvis", "clavicle", "bone", "muscle", "tendon", "ligament", "joint", "quad", "hamstring", "glute"
    ]
    pain_words = ["pain", "hurt", "injury", "sore", "ache", "stiff", "broken", "sprain", "tear", "surgery", "fracture"]

    found = []
    for token in doc:
        if token.lemma_ in body_parts or token.text in body_parts:
            start = max(0, token.i - 4)
            end = min(len(doc), token.i + 5)
            window = doc[start:end]
            if any(w.lemma_ in pain_words or w.text in pain_words for w in window):
                found.append(token.text)
            else:
                found.append(token.text) 
    return list(set(found))

def find_injury_solution(injury):
    if not df_injuries.empty and 'Injury_Location' in df_injuries.columns:
        match = df_injuries[
            df_injuries['Injury_Location'].str.contains(injury, case=False, na=False) |
            df_injuries['Condition_Name'].str.contains(injury, case=False, na=False)
        ]
        if not match.empty:
            return str(match.iloc[0]['Recommended_Modifications']), "injury"

    if not df_injuries.empty and 'Recommended_Modifications' in df_injuries.columns:
        random_rehab_row = df_injuries.sample(n=1).iloc[0]
        return str(random_rehab_row['Recommended_Modifications']), "rehab"

    return "Light Stretching | Walking", "fallback"

def get_allergy_data_from_dataset(allergy_word):
    bad_foods = [allergy_word.lower()]
    safe_alt = "Safe Alternative"
    
    if not df_food_dict.empty and 'Allergy' in df_food_dict.columns:
        matches = df_food_dict[df_food_dict['Allergy'].str.contains(allergy_word, case=False, na=False)]
        if not matches.empty:
            bad_foods.extend(matches['Food'].str.lower().tolist())

    if not df_allergies.empty and 'Foods_To_Strictly_Avoid' in df_allergies.columns:
        match = df_allergies[
            df_allergies['Condition_Name'].str.contains(allergy_word, case=False, na=False) |
            df_allergies['Foods_To_Strictly_Avoid'].str.contains(allergy_word, case=False, na=False)
        ]
        if not match.empty:
            foods_to_avoid_str = str(match.iloc[0]['Foods_To_Strictly_Avoid'])
            substitutions_str = str(match.iloc[0]['Recommended_Substitutions'])
            
            bad_foods.extend([f.strip().lower() for f in foods_to_avoid_str.split('|')])
            safe_alt = substitutions_str.split('|')[0].strip()
            
    return list(set(bad_foods)), safe_alt

# ==========================================
# 3. PYDANTIC MODELS
# ==========================================
class UserProfile(BaseModel):
    age: Optional[Any] = None
    weight: Optional[Any] = None
    height: Optional[Any] = None
    level: Optional[Any] = None
    goal: Optional[Any] = None
    injuries: Optional[Any] = None
    allergies: Optional[Any] = None
    days: Optional[Any] = "3 days"
    equipment: Optional[Any] = "full gym"

class AdaptRequest(BaseModel):
    weight: Optional[Any] = None       
    feedback: Optional[str] = None 
    days: Optional[Any] = None
    week_number: Optional[int] = 2    
    userStats: Optional[dict] = {}

class SwapRequest(BaseModel):
    request: str
    exerciseContext: str
    weight: Optional[Any] = "70"
    feedback: Optional[str] = "swap"
    days: Optional[Any] = "3"
    userStats: Optional[dict] = {}

# ==========================================
# 🚀 ROUTE 1: RECOMMENDATION (FIRST PLAN)
# ==========================================
@app.post("/api/recommend")
async def generate_plan(user: UserProfile):
    try:
        num_days = int(re.search(r'\d+', str(user.days)).group()) if user.days else 3
        weight_val = float(re.search(r'\d+', str(user.weight)).group()) if user.weight else 70.0
        daily_calories = int(weight_val * 24 * 1.5)

        detected_injuries = detect_injuries(user.injuries)

        X_w = np.hstack((
            mlb_goal.transform([[user.goal if user.goal else "General Fitness"]]),
            mlb_diff.transform([[user.level if user.level else "Beginner"]]),
            mlb_focus.transform([["Full Body"]]),
            mlb_equip.transform([[user.equipment if user.equipment else "bodyweight"]])
        ))
        
        _, idx_w = model_workouts.kneighbors(X_w, n_neighbors=15)
        workout_indices = list(idx_w[0])
        random.shuffle(workout_indices) 
        pool = df_workouts.iloc[workout_indices]

        workout_schedule = []
        workout_exercises = []
        workouts_list = [str(row.get('Routine', 'Basic exercise')) for _, row in pool.iterrows()]
        
        for i in range(num_days):
            day_name = f"Day {i+1} Workout"
            workout_schedule.append(day_name)
            
            raw_routine = workouts_list[i % len(workouts_list)]
            ex_list = [ex.strip() for ex in raw_routine.split('|') if ex.strip()]
            
            for inj in detected_injuries:
                rehab_text, _ = find_injury_solution(inj)
                rehab_ex_list = [ex.strip() for ex in rehab_text.split('|')]
                
                if len(ex_list) > 0 and len(rehab_ex_list) > 0:
                    ex_list[0] = f"🩹 REHAB: {rehab_ex_list[0]}"
                if len(ex_list) > 1 and len(rehab_ex_list) > 1:
                    ex_list[1] = f"🩹 REHAB: {rehab_ex_list[1]}"
            
            routines = []
            for ex in ex_list:
                is_rehab = "REHAB" in ex
                routines.append({
                    "name": ex,
                    "sets": 3,
                    "reps": "10-12" if not is_rehab else "15-20 (Slow)",
                    "description": f"Safety modification applied." if is_rehab else "Keep strict form."
                })
                
            workout_exercises.append({
                "day": day_name,
                "routines": routines
            })

        Xd = scaler_diet.transform([[30, 40, 30, daily_calories]])
        _, d_idx = model_diet.kneighbors(Xd, n_neighbors=15)
        diet_indices = list(d_idx[0])
        random.shuffle(diet_indices)
        diet_pool = df_diet.iloc[diet_indices]

        meals_pool = []
        spoken_allergy_words = [w for w in re.split(r'[^a-zA-Z]+', str(user.allergies).lower()) if len(w) > 3 and w not in ["have", "allergic", "severely", "very", "intolerant"]]

        for _, row in diet_pool.iterrows():
            meal = str(row.get('Meal_Name', 'Healthy Meal'))
            ingredients = str(row.get('Ingredients', "")).lower()

            for spoken_word in spoken_allergy_words:
                bad_foods_list, safe_alt = get_allergy_data_from_dataset(spoken_word)
                
                for bad_food in bad_foods_list:
                    raw_bad_food = bad_food.strip()
                    core_bad_food = raw_bad_food.split('(')[0].strip()

                    if core_bad_food in meal.lower() or core_bad_food in ingredients or spoken_word in meal.lower():
                        if raw_bad_food in meal.lower():
                            meal = re.sub(re.escape(raw_bad_food), safe_alt, meal, flags=re.IGNORECASE)
                        else:
                            meal = re.sub(core_bad_food, safe_alt, meal, flags=re.IGNORECASE)
                        
                        if spoken_word.endswith('s'):
                            meal = re.sub(spoken_word[:-1], safe_alt, meal, flags=re.IGNORECASE)
                            
                        meal += f" ⚠️ ({spoken_word.capitalize()}-Free)"
                        break 

            meals_pool.append(meal)

        diet_plans = []
        meal_idx = 0
        for i in range(num_days):
            day_meals = []
            for meal_type in ["Breakfast", "Lunch", "Dinner"]:
                food_item = meals_pool[meal_idx % len(meals_pool)]
                day_meals.append({"name": meal_type, "foods": [food_item]})
                meal_idx += 1
                
            diet_plans.append({"day": f"Day {i+1}", "meals": day_meals})

        return {
            "status": "success",
            "workoutPlan": {"schedule": workout_schedule, "exercises": workout_exercises},
            "dietPlan": {"dailyCalories": daily_calories, "dailyPlans": diet_plans},
            "injuries_detected": detected_injuries 
        }

    except Exception as e:
        print(f"Backend Error: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 🚀 ROUTE 2: WEEKLY CHECK-IN (ADAPT)
# ==========================================
@app.post("/api/adapt")
async def adapt_plan(req: AdaptRequest):
    try:
        user_stats = req.userStats or {}
        goal = user_stats.get("goal", "General Fitness")
        level = user_stats.get("level", "Beginner")
        equipment = user_stats.get("equipment", "bodyweight")
        allergies = user_stats.get("allergies", "none")
        injuries = user_stats.get("injuries", "none")
        week_num = req.week_number or 2

        num_days = int(re.search(r'\d+', str(req.days)).group()) if req.days else 3
        weight_val = float(re.search(r'\d+', str(req.weight)).group()) if req.weight else 70.0
        daily_calories = int(weight_val * 24 * 1.5)

        detected_injuries = detect_injuries(injuries)

        fb = str(req.feedback).lower()
        if "easy" in fb or "light" in fb:
            if level.lower() == "beginner": level = "Intermediate"
            elif level.lower() == "intermediate": level = "Advanced"

        X_w = np.hstack((
            mlb_goal.transform([[goal]]),
            mlb_diff.transform([[level]]),
            mlb_focus.transform([["Full Body"]]),
            mlb_equip.transform([[equipment]])
        ))
        
        _, idx_w = model_workouts.kneighbors(X_w, n_neighbors=15)
        workout_indices = list(idx_w[0])
        random.shuffle(workout_indices)
        pool = df_workouts.iloc[workout_indices]

        workout_schedule = []
        workout_exercises = []
        workouts_list = [str(row.get('Routine', 'Basic exercise')) for _, row in pool.iterrows()]
        
        for i in range(num_days):
            day_name = f"Week {week_num} - Day {i+1} Workout"
            workout_schedule.append(day_name)
            
            raw_routine = workouts_list[i % len(workouts_list)]
            ex_list = [ex.strip() for ex in raw_routine.split('|') if ex.strip()]
            
            for inj in detected_injuries:
                rehab_text, _ = find_injury_solution(inj)
                rehab_ex_list = [ex.strip() for ex in rehab_text.split('|')]
                
                if len(ex_list) > 0 and len(rehab_ex_list) > 0:
                    ex_list[0] = f"🩹 REHAB: {rehab_ex_list[0]}"
                if len(ex_list) > 1 and len(rehab_ex_list) > 1:
                    ex_list[1] = f"🩹 REHAB: {rehab_ex_list[1]}"
            
            routines = []
            for ex in ex_list:
                is_rehab = "REHAB" in ex
                routines.append({
                    "name": ex,
                    "sets": 3,
                    "reps": "10-12" if not is_rehab else "15-20 (Slow)",
                    "description": f"Safety modification applied." if is_rehab else "Keep strict form."
                })
                
            workout_exercises.append({
                "day": day_name,
                "routines": routines
            })

        Xd = scaler_diet.transform([[30, 40, 30, daily_calories]])
        _, d_idx = model_diet.kneighbors(Xd, n_neighbors=15)
        diet_indices = list(d_idx[0])
        random.shuffle(diet_indices)
        diet_pool = df_diet.iloc[diet_indices]

        meals_pool = []
        spoken_allergy_words = [w for w in re.split(r'[^a-zA-Z]+', str(allergies).lower()) if len(w) > 3 and w not in ["have", "allergic", "severely", "very", "intolerant"]]

        for _, row in diet_pool.iterrows():
            meal = str(row.get('Meal_Name', 'Healthy Meal'))
            ingredients = str(row.get('Ingredients', "")).lower()

            for spoken_word in spoken_allergy_words:
                bad_foods_list, safe_alt = get_allergy_data_from_dataset(spoken_word)
                for bad_food in bad_foods_list:
                    raw_bad_food = bad_food.strip()
                    core_bad_food = raw_bad_food.split('(')[0].strip()

                    if core_bad_food in meal.lower() or core_bad_food in ingredients or spoken_word in meal.lower():
                        if raw_bad_food in meal.lower():
                            meal = re.sub(re.escape(raw_bad_food), safe_alt, meal, flags=re.IGNORECASE)
                        else:
                            meal = re.sub(core_bad_food, safe_alt, meal, flags=re.IGNORECASE)
                        if spoken_word.endswith('s'):
                            meal = re.sub(spoken_word[:-1], safe_alt, meal, flags=re.IGNORECASE)
                        meal += f" ⚠️ ({spoken_word.capitalize()}-Free)"
                        break 
            meals_pool.append(meal)

        diet_plans = []
        meal_idx = 0
        for i in range(num_days):
            day_meals = []
            for meal_type in ["Breakfast", "Lunch", "Dinner"]:
                food_item = meals_pool[meal_idx % len(meals_pool)]
                day_meals.append({"name": meal_type, "foods": [food_item]})
                meal_idx += 1
                
            diet_plans.append({"day": f"Week {week_num} - Day {i+1}", "meals": day_meals})

        return {
            "status": "success",
            "updatedWorkoutPlan": {"schedule": workout_schedule, "exercises": workout_exercises},
            "updatedDietPlan": {"dailyCalories": daily_calories, "dailyPlans": diet_plans},
            "injuries_detected": detected_injuries 
        }

    except Exception as e:
        print(f"Adapt Error: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 🚀 ROUTE 3: QUICK SWAP EXERCISE
# ==========================================
@app.post("/api/swap")
async def swap_exercise(req: SwapRequest):
    try:
        user_stats = req.userStats or {}
        
        # 1. Combine saved injuries with whatever the user just said in the voice note!
        combined_text = f"{user_stats.get('injuries', '')} {req.request}"
        detected_injuries = detect_injuries(combined_text)

        # 2. Find a pool of exercises
        X_w = np.hstack((
            mlb_goal.transform([[user_stats.get("goal", "General Fitness")]]),
            mlb_diff.transform([[user_stats.get("level", "Beginner")]]),
            mlb_focus.transform([["Full Body"]]),
            mlb_equip.transform([[user_stats.get("equipment", "bodyweight")]])
        ))
        
        _, idx_w = model_workouts.kneighbors(X_w, n_neighbors=50)
        pool = df_workouts.iloc[idx_w[0]]
        
        all_exercises = []
        for _, row in pool.iterrows():
            routines = str(row.get('Routine', '')).split('|')
            for r in routines:
                if r.strip() and r.strip().lower() != req.exerciseContext.lower():
                    all_exercises.append(r.strip())
        
        random.shuffle(all_exercises)
        new_ex = all_exercises[0] if all_exercises else "Bodyweight Squats"
        is_rehab = False

        # 3. Apply Injury Replacements!
        for inj in detected_injuries:
            rehab_text, _ = find_injury_solution(inj)
            rehab_ex_list = [ex.strip() for ex in rehab_text.split('|')]
            if rehab_ex_list:
                new_ex = f"🩹 REHAB: {rehab_ex_list[0]}"
                is_rehab = True
                break

        # 4. Return exactly what React expects (as a "newExercise" object)
        return {
            "newExercise": {
                "name": new_ex,
                "sets": 3,
                "reps": "15-20 (Slow)" if is_rehab else "10-12",
                "description": f"Adjusted for safety: {', '.join(detected_injuries)}" if is_rehab else "Swapped based on your request."
            }
        }

    except Exception as e:
        print(f"Swap Error: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))