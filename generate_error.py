import csv
import random
import itertools

# --- CONFIGURATION ---
OUTPUT_FILENAME = "Synthetic_Error_Responses_Perfect_Distribution.csv"
TARGET_ROWS_PER_STAGE = 500

# --- DATA DICTIONARIES ---
PREFIXES = [
    "I didn't quite catch that.", "Sorry, I missed that.", "My apologies, I didn't understand.",
    "Could you repeat that?", "I'm not sure I follow.", "Wait, say that again?",
    "Hmm, that didn't compute.", "Let's try that one more time.", "I think I misheard you.",
    "[Name], I didn't quite get that.", "Sorry [Name], could you clarify?", "Oops, missed that."
]

SUFFIXES = [
    "Let's try again.", "What do you say?", "Can you clarify?", "Let me know.",
    "Take your time.", "Any ideas?", "Just say the word.", "I'm ready when you are."
]

STAGE_CORES = {
    "GREETING": [
        "Are you ready to crush some goals?",
        "Are we ready to get started?",
        "Let me know if you want to begin.",
        "Just say yes if you are ready to train."
    ],
    "CONSENT": [
        "I just need a simple yes or no to collect your fitness data.",
        "Please confirm with a yes or no so we can proceed safely.",
        "Do I have your permission to use this data for your plan? Yes or no?",
        "I need your consent to continue. Is that a yes?"
    ],
    "INTRO": [ # Age
        "Could you tell me your age in numbers?",
        "How many years old are you?",
        "I just need a valid number for your age.",
        "Please give me your exact age."
    ],
    "AGE": [ # Weight (Saved after Age)
        "How much do you weigh? Just the number is fine.",
        "I need a valid number for your weight.",
        "Could you tell me your current weight?",
        "Please specify your weight in numbers."
    ],
    "HEIGHT": [
        "I need a clear number for your height.",
        "How tall are you? Please use numbers.",
        "Could you repeat your height for me?",
        "Just give me your height in feet/inches or cm."
    ],
    "GOAL": [
        "Are you trying to lose weight, build muscle, or something else?",
        "What's the main focus? Fat loss, muscle gain, or endurance?",
        "I need to know your primary fitness goal to build the plan.",
        "Could you be a bit more specific about your main goal?"
    ],
    "LEVEL": [
        "Are you a beginner, intermediate, or advanced?",
        "Please just tell me if you are beginner, intermediate, or advanced.",
        "I need your experience level: beginner, intermediate, or advanced.",
        "Which level fits you best: beginner, intermediate, or advanced?"
    ],
    "DAYS": [
        "How many days a week can you train? Choose between 1 and 7.",
        "I need a number of days you can commit to working out.",
        "Please give me a number from 1 to 7 for your workout days.",
        "How many days per week? Just a number, please."
    ],
    "EQUIPMENT": [
        "Will you be at a full gym, at home, or doing bodyweight?",
        "What equipment do you have? Gym, dumbbells, or none?",
        "I need to know your setup: gym, home equipment, or bodyweight.",
        "Please tell me where you'll be working out."
    ],
    "ALLERGIES_CHECK": [
        "If you have allergies, please list them. If not, just say none.",
        "Do you have any dietary restrictions? Yes or no?",
        "I didn't catch that. Any food allergies I should know about?",
        "Please clearly state any allergies, or say 'none'."
    ],
    "INJURIES_CHECK": [
        "If you have injuries, please describe them. If not, say none.",
        "Any physical limitations or injuries I need to work around?",
        "Please clearly state any injuries, or just tell me 'no'.",
        "I need to know if you have any injuries to keep the plan safe."
    ]
}

def generate_dataset():
    data = []
    
    for stage, cores in STAGE_CORES.items():
        # Generate all possible combinations for this stage
        combinations = list(itertools.product(PREFIXES, cores, SUFFIXES))
        
        # Shuffle to randomize the order
        random.shuffle(combinations)
        
        # Take up to the target number of rows to avoid millions of rows
        selected_combinations = combinations[:TARGET_ROWS_PER_STAGE]
        
        for prefix, core, suffix in selected_combinations:
            # Randomly decide to drop the suffix sometimes for more natural phrasing
            if random.random() > 0.5:
                full_message = f"{prefix} {core} {suffix}"
            else:
                full_message = f"{prefix} {core}"
                
            data.append({"stage": stage, "full_error_message": full_message})

    # Write to CSV
    with open(OUTPUT_FILENAME, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["stage", "full_error_message"])
        writer.writeheader()
        writer.writerows(data)
        
    print(f"✅ Successfully generated {len(data)} unique error responses across {len(STAGE_CORES)} stages.")
    print(f"📁 Saved to: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    generate_dataset()