"""
Pressing Analyst page.
Follows the same framework pattern as football_scout.py:
  1. Load all data → PressingStats (Z-scores computed against league)
  2. Sidebar selector → PressingTeam data point
  3. PressingDistributionPlot → league distribution + selected team
  4. PressingDescription → wordalization via LLM
  5. PressingChat → conversational follow-up
"""

import os

import pandas as pd
import streamlit as st

from classes.data_source import PressingStats
from classes.data_point import PressingTeam
from classes.visual import PressingDistributionPlot
from classes.description import PressingDescription
from classes.chat import PressingChat

from utils.page_components import add_common_page_elements
from utils.utils import create_chat

# ── Constants ─────────────────────────────────────────────────────────────────
_LOCAL_DATA_PATH = "/Users/hugovicente/Documents/CODING/V2_Twelve_wordalisation/databases/dynamic_events_pl_24"
DATA_PATH = _LOCAL_DATA_PATH if os.path.exists(_LOCAL_DATA_PATH) else "data/dynamic_events_pl_24"

PRESSING_COLS = [
    "match_id", "team_id",
    "pressing_chain", "pressing_chain_length", "pressing_chain_end_type",
    "pressing_chain_index",
    "stop_possession_danger", "force_backward",
    "beaten_by_possession", "beaten_by_movement",
    "lead_to_shot",
]

# ── Page setup ─────────────────────────────────────────────────────────────────
sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()
st.markdown("## Pressing Analyst")


# ── Cached data loading ────────────────────────────────────────────────────────
@st.cache_data
def load_reference_data():
    teams = pd.read_parquet(f"{DATA_PATH}/teams.parquet")
    matches = pd.read_parquet(f"{DATA_PATH}/matches.parquet")
    return teams, matches


@st.cache_data
def load_all_pressing_events(match_ids: tuple) -> pd.DataFrame:
    frames = []
    for mid in match_ids:
        path = f"{DATA_PATH}/dynamic/{mid}.parquet"
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path, columns=PRESSING_COLS)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


try:
    teams_df, matches_df = load_reference_data()
except FileNotFoundError:
    st.info(
        "The Pressing Analyst requires match data that is not bundled with this deployment. "
        "Run the app locally to access the full pressing analysis."
    )
    st.stop()

all_match_ids = tuple(matches_df["id"].tolist())

with st.spinner("Loading league pressing data…"):
    df_all_events = load_all_pressing_events(all_match_ids)

if df_all_events.empty:
    st.error("No pressing data found. Check DATA_PATH.")
    st.stop()


# ── Build league-wide PressingStats ────────────────────────────────────────────
pressing_stats = PressingStats(df_events=df_all_events, teams_df=teams_df)
pressing_stats.calculate_statistics(
    metrics=PressingStats.METRICS,
    negative_metrics=PressingStats.NEGATIVE_METRICS,
)

team_names = sorted(pressing_stats.df["team_name"].tolist())


# ── Session state: track explicit team selection ───────────────────────────────
if "pressing_team_selected" not in st.session_state:
    st.session_state.pressing_team_selected = False


def on_team_change():
    st.session_state.pressing_team_selected = True


# ── Sidebar: team dropdown ─────────────────────────────────────────────────────
with sidebar_container:
    selected_team_name = st.selectbox(
        "Select a team",
        team_names,
        index=None,
        placeholder="Choose a team…",
        key="pressing_team_name",
        on_change=on_team_change,
    )

# ── Welcome screen (shown until user picks a team) ─────────────────────────────
if not st.session_state.pressing_team_selected:
    with st.chat_message("assistant", avatar="data/ressources/img/twelve_chat_logo.svg"):
        st.markdown("Which Premier League team would you like to analyse?")
        st.markdown(
            "Select a team from the **menu on the left** and I'll show you their pressing profile — "
            "including a league distribution, an AI summary, and a chat where you can dig deeper."
        )
    st.stop()


# ── Build PressingTeam data point for the selected team ───────────────────────
team_row = pressing_stats.df[
    pressing_stats.df["team_name"] == selected_team_name
].iloc[0]
ser_metrics = team_row.drop(labels=["team_id", "team_name"])
team = PressingTeam(
    id=team_row["team_id"],
    name=selected_team_name,
    ser_metrics=ser_metrics,
    relevant_metrics=PressingStats.METRICS,
)


# ── Chat state hash ────────────────────────────────────────────────────────────
to_hash = (team.id, "pressing_analyst")
chat = create_chat(to_hash, PressingChat, team, pressing_stats)

if chat.state == "empty":

    visual = PressingDistributionPlot(
        PressingStats.METRICS[::-1],
        plot_type="pressing",
    )
    visual.add_title_from_team(team)
    visual.add_teams(pressing_stats, metrics=PressingStats.METRICS)
    visual.add_team(team, n_group=len(pressing_stats.df), metrics=PressingStats.METRICS)

    description = PressingDescription(team)
    summary = description.stream_gpt(stream=True)

    chat.add_message(
        f"Please summarise {team.name}'s pressing for me.",
        role="user",
        user_only=False,
        visible=False,
    )
    chat.add_message(visual)
    chat.add_message(summary)
    chat.state = "default"

chat.get_input()
chat.display_messages()
chat.save_state()
