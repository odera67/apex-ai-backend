import pandas as pd
import random

print("Generating Synthetic User Responses Dataset...")

user_data = []
# Added GREETING, CONSENT, and CONFIRMATION to match your complete flow!
stages_for_users = [
    "GREETING", "CONSENT", "AGE", "WEIGHT", "HEIGHT", "GOAL", 
    "LEVEL", "DAYS", "EQUIPMENT", "ALLERGIES", "INJURIES", "CONFIRMATION"
]

for _ in range(5500):
    stage = random.choice(stages_for_users)
    raw_text = ""
    extracted = ""
    
    if stage == "GREETING":
        readiness = random.choice(["yes", "no"])
        if readiness == "yes":
            templates = ["Yes I am", "Let's do it", "Ready", "I'm ready", "Yeah let's go", "Absolutely", "Yes"]
        else:
            templates = ["No", "Not yet", "Give me a minute", "Nah", "Not ready"]
        raw_text = random.choice(templates)
        extracted = readiness

    elif stage == "CONSENT":
        consent = random.choice(["yes", "no"])
        if consent == "yes":
            templates = ["Yes", "I consent", "Sure", "Okay", "Yeah that's fine", "Go ahead", "Yes you do"]
        else:
            templates = ["No", "I don't consent", "Nope", "I would rather not", "Nah"]
        raw_text = random.choice(templates)
        extracted = consent

    elif stage == "AGE":
        age = random.randint(16, 65)
        templates = ["I am {} years old", "{}", "I'm {}", "I just turned {}", "I am {}"]
        raw_text = random.choice(templates).format(age)
        extracted = str(age)

    elif stage == "WEIGHT":
        weight = random.randint(100, 300)
        templates = ["I weigh {} lbs", "{} pounds", "{}", "About {}", "I am {} lbs", "Around {}"]
        raw_text = random.choice(templates).format(weight)
        extracted = str(weight)

    elif stage == "HEIGHT":
        cm = random.randint(150, 195)
        templates = ["{} cm", "I am {} centimeters", "{}", "About {} cm"]
        raw_text = random.choice(templates).format(cm)
        extracted = str(cm)

    elif stage == "GOAL":
        goals = ["lose weight", "build muscle", "get shredded", "endurance", "burn fat", "tone up"]
        goal = random.choice(goals)
        templates = ["I want to {}", "My goal is to {}", "{}", "Looking to {}", "Mainly to {}"]
        raw_text = random.choice(templates).format(goal)
        extracted = goal

    elif stage == "LEVEL":
        levels = ["beginner", "intermediate", "advanced", "pro", "novice"]
        level = random.choice(levels)
        templates = ["I am a {}", "{}", "Pretty much a {}", "I'd say {}", "Just a {}"]
        raw_text = random.choice(templates).format(level)
        extracted = level

    elif stage == "DAYS":
        days = random.randint(2, 6)
        templates = ["{} days", "I can do {} days a week", "{}", "About {} days", "I have {} days free"]
        raw_text = random.choice(templates).format(days)
        extracted = str(days)

    elif stage == "EQUIPMENT":
        equipments = ["full gym", "dumbbells", "bodyweight", "home gym", "kettlebells", "resistance bands"]
        equip = random.choice(equipments)
        templates = ["I have access to {}", "Just {}", "{}", "I workout at a {}", "I only have {}"]
        raw_text = random.choice(templates).format(equip)
        extracted = equip

    elif stage == "ALLERGIES":
        allergies = ["none", "peanut", "lactose intolerant", "gluten", "vegan", "shellfish"]
        allergy = random.choice(allergies)
        if allergy == "none":
            templates = ["Nope", "None", "No allergies", "I eat everything", "Nah"]
        else:
            templates = ["I am {}", "I'm allergic to {}s", "{}", "I have a {} allergy"]
        raw_text = random.choice(templates).format(allergy)
        extracted = allergy

    elif stage == "INJURIES":
        injuries = ["none", "bad knee", "lower back pain", "shoulder impingement", "twisted ankle", "tennis elbow"]
        injury = random.choice(injuries)
        if injury == "none":
            templates = ["No", "None", "I'm perfectly healthy", "No injuries", "Nope"]
        else:
            templates = ["I have a {}", "My {} hurts sometimes", "{}", "Dealing with {}"]
        raw_text = random.choice(templates).format(injury)
        extracted = injury

    elif stage == "CONFIRMATION":
        confirmation = random.choice(["yes", "no"])
        if confirmation == "yes":
            templates = ["Yes", "Looks good", "Perfect", "That's right", "Generate it", "Go ahead", "Looks correct"]
        else:
            templates = ["No", "Wait", "I need to change something", "That is wrong", "Incorrect"]
        raw_text = random.choice(templates)
        extracted = confirmation

    user_data.append({"stage": stage, "user_raw_input": raw_text, "extracted_value": extracted})

df_users = pd.DataFrame(user_data)
df_users.to_csv("Synthetic_User_Responses_5000.csv", index=False)

print(f"✅ Created Synthetic_User_Responses_5000.csv with {len(df_users)} rows!")