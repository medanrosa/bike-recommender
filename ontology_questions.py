# each tuple is: (field_name, prompt, {str→int} mapping or None if free numeric)
QUESTIONS = [
    ("experience_level",
     "What's your riding experience? (beginner/intermediate/expert): ",
     {"beginner": 0, "intermediate": 1, "expert": 2}
    ),
    ("preferred_style",
     "Which style do you prefer? (supersport/supermoto/naked/touring/dirtbike): ",
     {"supersport": 0, "supermoto": 1, "naked": 2, "touring": 3, "dirtbike": 4}
    ),
    ("geography",
     "Where do you ride most? (mountain/coastal/city/small_town/highway/back_roads): ",
     {"mountain": 0, "coastal": 1, "city": 2, "small_town": 3, "highway": 4, "back_roads": 5}
    ),
    ("budget",
     "Enter your maximum budget (USD): ",
     None
    )
]

def ask_user() -> dict:
    answers = {}
    for field, prompt, mapping in QUESTIONS:
        while True:
            ans = input(prompt).strip().lower()
            if mapping:
                if ans in mapping:
                    answers[field] = mapping[ans]
                    break
                else:
                    print(f"↳ please choose one of {list(mapping.keys())}")
            else:
                try:
                    answers[field] = int(ans)
                    break
                except ValueError:
                    print("↳ please enter a whole‐number budget")
    return answers