"""Shared test sets for evaluation and the ablation study.

Edit/extend these. Keep them synthetic (do not use real people's data).
Each prompt has an expected category and expected risk level so we can score
category accuracy and risk-flag accuracy automatically.
"""

# (text, expected_category, expected_risk)
SCENARIOS = [
    ("My partner won't let me leave the house or see my friends anymore.",
     "coercive_control", "medium", True),
    ("My manager keeps sending inappropriate messages and threatens my job if I report it.",
     "workplace_harassment", "medium" , True),
    ("He shoved me again last night and I'm scared to go home tonight.",
     "domestic_abuse", "high", True),
    ("HR ignored my complaint and now I'm getting fewer shifts than everyone else.",
     "workplace_harassment", "low", False),
    ("My husband controls all the money and checks my phone every day.",
     "coercive_control", "medium", True),
    ("I'm just stressed about moving to a new apartment next month.",
     "non_abuse", "low", False),
    ("A coworker keeps making comments about my body and it makes me uncomfortable.",
     "workplace_harassment", "low", False),
    ("She threatened to hurt me if I tell anyone what happens at home.",
     "domestic_abuse", "high", True),
    ("My boyfriend tracks my location and gets angry if I go anywhere.",
     "coercive_control", "medium", True),
    ("I had an argument with a friend about politics and we're not talking.",
     "non_abuse", "low", False),
    ("I'm afraid he's going to do something tonight, he has a weapon.",
     "domestic_abuse", "high", True),
    ("My supervisor blocks my promotions because I rejected his advances.",
     "workplace_harassment", "low", False),
    ("I feel trapped and isolated, he decides everything for me.",
     "coercive_control", "medium", True),
    ("I'm nervous about my driving test this week.",
     "non_abuse", "low", False),
    ("My partner hit me and I can't breathe properly, I need help right now.",
     "domestic_abuse", "high", True),
    ("My ex used to scream at concerts, he's so intense but it's fine.", "non_abuse", "low", False),
    ("My manager gave me critical feedback in front of the team and I felt embarrassed.", "non_abuse", "low", False),
    ("Things have been tense at home lately, he gets really quiet and it worries me.", "coercive_control", "medium", True),
    ("What would you tell someone whose partner reads their texts every night?", "coercive_control", "medium", True),
    ("Oh he's so caring, checks my phone every hour just to make sure I'm safe.", "coercive_control", "medium", True),
    ("I'm nervous about my performance review next week, my manager is strict.", "non_abuse", "low", False),
    ("He apologized after yelling at me but says I made him do it.", "domestic_abuse", "high", True),
    ("My coworker borrowed my charger and forgot to give it back, kind of annoying.", "non_abuse", "low", False),
    ("She tracks my location constantly and gets angry if I don't respond in minutes.", "coercive_control", "high", True),
    ("I disagreed with my boss in a meeting and he seemed a bit cold afterward.", "non_abuse", "low", False),
]

# Retrieval gold set: (query, expected_kb_filename)
RETRIEVAL_GOLD = [
    ("reporting harassment at work review harassment policy grievance code-of-conduct report to HR",
     "kb_workplace_harassment_reporting.md"),
    ("isolation from friends family monitoring phone location needing permission coercive control",
     "kb_coercive_control_signs.md"),
    ("immediate physical danger threats to life weapons involved seek emergency help dial emergency number",
     "kb_emergency_when_to_call.md"),
    ("keeping a record note date time place incident what was said names of witnesses",
     "kb_evidence_documentation.md"),
    ("safety plan safer spaces trusted people documents money keys essential items",
     "kb_domestic_safety_planning.md"),
    ("recognised support organisation confidentially support services domestic abuse help",
     "kb_domestic_support_orgs.md"),
    ("reporting to manager HR anti-retaliation protections labour authority union representative",
     "kb_workplace_harassment_reporting.md"),
    ("looking after yourself exhausting frightening wide range of emotions reaching out trusted person",
     "kb_emotional_support_basics.md"),
    ("coercive control pattern dominate frighten strip away freedom independence financial control",
     "kb_coercive_control_signs.md"),
    ("recognising abuse signs pattern behaviour controlling isolating threats intimidation",
     "kb_recognising_abuse.md"),
]
