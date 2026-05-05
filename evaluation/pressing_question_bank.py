"""
Question bank for testing PressingChat tool routing.

Each entry has:
  q            — the user question
  expected     — the tool that should be called, or None for out-of-scope (decline)

TEAM_NAME is the mock selected team used in the system prompt.
"""

TEAM_NAME = "Liverpool"

QUESTIONS = [
    # ── get_team_pressing_summary ─────────────────────────────────────────────
    # Questions about the selected team's own pressing profile
    {"q": "How does Liverpool press?",                                   "expected": "get_team_pressing_summary"},
    {"q": "What are their main pressing strengths?",                     "expected": "get_team_pressing_summary"},
    {"q": "Give me an overview of their pressing style.",                "expected": "get_team_pressing_summary"},
    {"q": "What are Liverpool's pressing weaknesses?",                   "expected": "get_team_pressing_summary"},
    {"q": "How effective is their press overall?",                       "expected": "get_team_pressing_summary"},
    {"q": "Can you summarise their pressing data?",                      "expected": "get_team_pressing_summary"},
    {"q": "What makes their pressing distinctive in the Premier League?","expected": "get_team_pressing_summary"},
    {"q": "How do they perform when pressing high up the pitch?",        "expected": "get_team_pressing_summary"},
    {"q": "What's their pressing profile this season?",                  "expected": "get_team_pressing_summary"},
    {"q": "Tell me about how Liverpool win the ball back.",               "expected": "get_team_pressing_summary"},

    # ── search_pressing_knowledge ─────────────────────────────────────────────
    # Conceptual / tactical questions not tied to specific team numbers
    {"q": "What does beaten rate mean?",                                 "expected": "search_pressing_knowledge"},
    {"q": "How is recovery rate calculated?",                            "expected": "search_pressing_knowledge"},
    {"q": "What is a pressing chain?",                                   "expected": "search_pressing_knowledge"},
    {"q": "What makes a press tactically effective?",                    "expected": "search_pressing_knowledge"},
    {"q": "What does it mean to force a team to play backwards?",        "expected": "search_pressing_knowledge"},
    {"q": "What does danger rate tell us about a team's pressing?",      "expected": "search_pressing_knowledge"},
    {"q": "Why is converting pressing regains into shots important?",    "expected": "search_pressing_knowledge"},
    {"q": "What's the difference between a high press and a mid-block?", "expected": "search_pressing_knowledge"},
    {"q": "How do coaches typically set up a pressing trap?",            "expected": "search_pressing_knowledge"},
    {"q": "What does force backward rate measure?",                      "expected": "search_pressing_knowledge"},

    # ── compare_team_pressing ─────────────────────────────────────────────────
    # Questions explicitly comparing the selected team to a named other team
    {"q": "How do Liverpool compare to Arsenal in pressing?",            "expected": "compare_team_pressing"},
    {"q": "Are they better pressers than Manchester City?",              "expected": "compare_team_pressing"},
    {"q": "Compare Liverpool's pressing to Chelsea's.",                  "expected": "compare_team_pressing"},
    {"q": "Who presses more effectively, Liverpool or Everton?",         "expected": "compare_team_pressing"},
    {"q": "Is their beaten rate better or worse than Brentford's?",      "expected": "compare_team_pressing"},
    {"q": "How does their recovery rate compare to Manchester United?",  "expected": "compare_team_pressing"},
    {"q": "Put Liverpool up against Tottenham on pressing metrics.",     "expected": "compare_team_pressing"},
    {"q": "Are they more aggressive pressers than Wolves?",              "expected": "compare_team_pressing"},
    {"q": "Compare their pressing intensity with Brighton.",             "expected": "compare_team_pressing"},
    {"q": "How does Liverpool's press hold up against Newcastle?",       "expected": "compare_team_pressing"},

    # ── get_league_rankings ───────────────────────────────────────────────────
    # Questions about league position / standings for pressing metrics
    {"q": "Where do Liverpool rank in the league for pressing?",         "expected": "get_league_rankings"},
    {"q": "How do they compare to the rest of the Premier League?",      "expected": "get_league_rankings"},
    {"q": "Are they top 5 for recovery rate?",                           "expected": "get_league_rankings"},
    {"q": "Where do they sit in the league table for pressing?",         "expected": "get_league_rankings"},
    {"q": "What's their league position for beaten rate?",               "expected": "get_league_rankings"},
    {"q": "Are they above or below average for lead-to-shot rate?",      "expected": "get_league_rankings"},
    {"q": "How do they rank for danger rate across the league?",         "expected": "get_league_rankings"},
    {"q": "Show me their standings for all pressing metrics.",           "expected": "get_league_rankings"},
    {"q": "What percentile are they in for forcing teams backwards?",    "expected": "get_league_rankings"},
    {"q": "Are they the best pressing team in the Premier League?",      "expected": "get_league_rankings"},

    # ── out_of_scope ──────────────────────────────────────────────────────────
    # Should be declined — no tool called
    {"q": "Who will win the Premier League this season?",                "expected": None},
    {"q": "How many goals has Erling Haaland scored?",                   "expected": None},
    {"q": "What's the best formation in modern football?",               "expected": None},
    {"q": "Tell me about Arsenal's attacking play.",                     "expected": None},
    {"q": "Can you predict the result of next week's match?",            "expected": None},
    {"q": "What's the weather like in Liverpool?",                       "expected": None},
    {"q": "Write me a poem about pressing football.",                    "expected": None},
    {"q": "Who is the best manager in the Premier League?",              "expected": None},
    {"q": "How do I improve my own pressing as a player?",               "expected": None},
    {"q": "What's Klopp's tactical philosophy?",                         "expected": None},
]
