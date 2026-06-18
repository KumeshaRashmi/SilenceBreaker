"""Shared test sets for evaluation and the ablation study.

Edit/extend these. Keep them synthetic (do not use real people's data).
Each prompt has an expected category and expected risk level so we can score
category accuracy and risk-flag accuracy automatically.
"""

# (text, expected_category, expected_risk)
SCENARIOS = [
    ("My partner won't let me leave the house or see my friends anymore.",
     "coercive_control", "medium"),
    ("My manager keeps sending inappropriate messages and threatens my job if I report it.",
     "workplace_harassment", "medium"),
    ("He shoved me again last night and I'm scared to go home tonight.",
     "domestic_abuse", "high"),
    ("HR ignored my complaint and now I'm getting fewer shifts than everyone else.",
     "workplace_harassment", "low"),
    ("My husband controls all the money and checks my phone every day.",
     "coercive_control", "medium"),
    ("I'm just stressed about moving to a new apartment next month.",
     "non_abuse", "low"),
    ("A coworker keeps making comments about my body and it makes me uncomfortable.",
     "workplace_harassment", "low"),
    ("She threatened to hurt me if I tell anyone what happens at home.",
     "domestic_abuse", "high"),
    ("My boyfriend tracks my location and gets angry if I go anywhere.",
     "coercive_control", "medium"),
    ("I had an argument with a friend about politics and we're not talking.",
     "non_abuse", "low"),
    ("I'm afraid he's going to do something tonight, he has a weapon.",
     "domestic_abuse", "high"),
    ("My supervisor blocks my promotions because I rejected his advances.",
     "workplace_harassment", "low"),
    ("I feel trapped and isolated, he decides everything for me.",
     "coercive_control", "medium"),
    ("I'm nervous about my driving test this week.",
     "non_abuse", "low"),
    ("My partner hit me and I can't breathe properly, I need help right now.",
     "domestic_abuse", "high"),
]

# Retrieval gold set: (query, expected_kb_filename)
RETRIEVAL_GOLD = [
    ("how do I report harassment at work",
     "kb_workplace_harassment_reporting.md"),
    ("my partner controls my money and won't let me see friends",
     "kb_coercive_control_signs.md"),
    ("I think I'm in danger tonight and need urgent help",
     "kb_emergency_when_to_call.md"),
    ("how to keep a record of incidents as evidence",
     "kb_evidence_documentation.md"),
    ("what is a safety plan for leaving an abusive home",
     "kb_domestic_safety_planning.md"),
    ("where can I find support organisations for domestic abuse",
     "kb_domestic_support_orgs.md"),
    ("my rights at work regarding harassment",
     "kb_workplace_rights_overview.md"),
    ("how do I cope emotionally with what is happening",
     "kb_emotional_support_basics.md"),
    ("what counts as coercive controlling behaviour",
     "kb_coercive_control_signs.md"),
    ("signs that a relationship is abusive",
     "kb_recognising_abuse.md"),
]
