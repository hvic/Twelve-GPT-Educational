"""
Routing accuracy test for PressingChat.

Tests whether the LLM selects the correct tool for each question in the
question bank, without needing Streamlit or real pressing data.

Run from the project root:
    python evaluation/test_pressing_routing.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

# ── Resolve project root so we can find secrets ───────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.pressing_question_bank import QUESTIONS, TEAM_NAME

# ── Load secrets directly (avoids st.secrets dependency) ─────────────────────
def _load_secrets():
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    secrets = {}
    with open(secrets_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                secrets[key.strip()] = value.strip().strip('"').strip("'")
    return secrets

secrets = _load_secrets()
GPT_KEY = secrets["GPT_KEY"]
GPT_BASE = secrets["GPT_BASE"]
GPT_CHAT_MODEL = secrets["GPT_CHAT_MODEL"]

# ── Tool definitions (copied from PressingChat.TOOLS) ────────────────────────
TOOLS = [
    {
        "type": "function",
        "name": "get_team_pressing_summary",
        "description": (
            "Returns a detailed statistical summary and wordalization of the selected team's "
            "pressing performance. Use this when the user asks about the selected team's pressing "
            "style, strengths, weaknesses, or overall profile."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "search_pressing_knowledge",
        "description": (
            "Searches the pressing tactics knowledge base to answer general conceptual questions "
            "about pressing — e.g. what metrics mean, what makes a good press, tactical concepts. "
            "Use this for questions not specific to the selected team's numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's question to search for in the knowledge base.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "compare_team_pressing",
        "description": (
            "Compares the selected team's pressing statistics side-by-side with another "
            "Premier League team. Use this when the user asks how the selected team compares "
            "to or differs from a specific other team."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "other_team_name": {
                    "type": "string",
                    "description": "The name of the Premier League team to compare against.",
                }
            },
            "required": ["other_team_name"],
        },
    },
    {
        "type": "function",
        "name": "get_league_rankings",
        "description": (
            "Returns the selected team's rank in the Premier League for each pressing metric, "
            "showing where they stand relative to all other teams. Use this when the user asks "
            "about league position, rankings, or how the team compares to the rest of the league."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

# ── Instruction messages (mirrors PressingChat.instruction_messages) ──────────
def build_instruction_messages(team_name):
    return [
        {
            "role": "system",
            "content": (
                "You are a UK-based tactical football analyst specialising in pressing and "
                "out-of-possession play in the Premier League. "
                "You ONLY answer questions about pressing tactics and the pressing statistics "
                "available through your tools. "
                "If a user asks about an unrelated topic (other sports, politics, etc.), "
                "politely decline and refocus on pressing analysis. "
                "If a user asks about a team that is not currently selected, do NOT answer about that team — "
                "politely decline and tell them to use the sidebar to switch teams."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The user has selected {team_name}. The conversation focuses on their pressing. "
                "You have four tools: get_team_pressing_summary, search_pressing_knowledge, "
                "compare_team_pressing, and get_league_rankings. "
                "Always use a tool to ground your answer in data. "
                "Give concise 2-3 sentence responses directly to the user."
            ),
        },
    ]

# ── Run routing test ──────────────────────────────────────────────────────────
def get_tool_call(client, messages, question):
    messages_with_q = messages + [{"role": "user", "content": question}]
    response = client.responses.create(
        model=GPT_CHAT_MODEL,
        input=messages_with_q,
        tools=TOOLS,
        tool_choice="auto",
    )
    tool_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
    return tool_calls[0].name if tool_calls else None


def run():
    client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)
    messages = build_instruction_messages(TEAM_NAME)

    results = []
    by_category = defaultdict(list)

    print(f"\n=== Pressing Chat Routing Test  (team: {TEAM_NAME}) ===\n")

    for item in QUESTIONS:
        question = item["q"]
        expected = item["expected"]
        category = expected if expected else "out_of_scope"

        actual = get_tool_call(client, messages, question)
        passed = actual == expected

        results.append({"question": question, "expected": expected, "actual": actual, "pass": passed})
        by_category[category].append(passed)

        status = "✓" if passed else "✗"
        print(f"  {status}  [{category}]")
        if not passed:
            print(f"      Q: {question}")
            print(f"      Expected: {expected}  →  Got: {actual}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(results)
    passed_total = sum(r["pass"] for r in results)

    print(f"\n{'─'*55}")
    print(f"{'Category':<35} {'Score':>8}   {'Acc':>6}")
    print(f"{'─'*55}")
    for cat, outcomes in sorted(by_category.items()):
        n = len(outcomes)
        p = sum(outcomes)
        print(f"  {cat:<33} {p:>4}/{n:<4}   {p/n*100:>5.1f}%")
    print(f"{'─'*55}")
    print(f"  {'OVERALL':<33} {passed_total:>4}/{total:<4}   {passed_total/total*100:>5.1f}%")
    print()

    # ── Save full results ─────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "pressing_routing_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {out_path}\n")


if __name__ == "__main__":
    run()
