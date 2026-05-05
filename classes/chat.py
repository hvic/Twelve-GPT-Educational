import streamlit as st
from openai import OpenAI
from itertools import groupby
from types import GeneratorType
import pandas as pd
import json

from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import (
        GPT_BASE,
        GPT_KEY,
        GPT_CHAT_MODEL,
        GPT_SUPPORTS_REASONING,
        GPT_AVAILABLE_REASONING_EFFORTS,
        GPT_SUPPORTS_TEMPERATURE,
    )

from classes.description import (
    PlayerDescription,
    CountryDescription,
    PersonDescription,
    PressingDescription,
)
from classes.embeddings import PlayerEmbeddings, CountryEmbeddings, PersonEmbeddings, PressingEmbeddings

from classes.visual import Visual, DistributionPlot, DistributionPlotPersonality

import utils.sentences as sentences
from utils.gemini import convert_messages_format


class Chat:
    function_names = []

    def __init__(self, chat_state_hash, state="empty"):

        if (
            "chat_state_hash" not in st.session_state
            or chat_state_hash != st.session_state.chat_state_hash
        ):
            # st.write("Initializing chat")
            st.session_state.chat_state_hash = chat_state_hash
            st.session_state.messages_to_display = []
            st.session_state.chat_state = state
        if isinstance(self, PlayerChat):
            self.name = self.player.name
        elif isinstance(self, PersonChat):
            self.name = self.person.name
        else:
            pass

        # Set session states as attributes for easier access
        self.messages_to_display = st.session_state.messages_to_display
        self.state = st.session_state.chat_state

    def instruction_messages(self):
        """
        Sets up the instructions to the agent. Should be overridden by subclasses.
        """
        return []

    def add_message(self, content, role="assistant", user_only=True, visible=True):
        """
        Used by app.py to start off the conversation with plots and descriptions.
        """
        message = {"role": role, "content": content}
        self.messages_to_display.append(message)

    # def get_input(self):
    #     """
    #     Get input from streamlit."""

    #     if x := st.chat_input(
    #         placeholder=f"What else would you like to know about {self.player.name}?"
    #     ):
    #         if len(x) > 500:
    #             st.error(
    #                 f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
    #             )

    #         self.handle_input(x)

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        """
        The main function that calls the GPT-4 API and processes the response.
        """

        # Get the instruction messages.
        messages = self.instruction_messages()

        # Add a copy of the user messages. This is to give the assistant some context.
        messages = messages + self.messages_to_display.copy()

        # Get relevant information from the user input and then generate a response.
        # This is not added to messages_to_display as it is not a message from the assistant.
        get_relevant_info = self.get_relevant_info(input)

        # Now add the user input to the messages. Don't add system information and system messages to messages_to_display.
        self.messages_to_display.append({"role": "user", "content": input})

        messages.append(
            {
                "role": "user",
                "content": f"Here is the relevant information to answer the users query: {get_relevant_info}\n\n```User: {input}```",
            }
        )

        # Remove all items in messages where content is not a string
        messages = [
            message for message in messages if isinstance(message["content"], str)
        ]

        # Show the messages in an expander
        st.expander("Chat transcript", expanded=False).write(messages)

        # Check if use gemini is set to true
        if USE_GEMINI:
            import google.generativeai as genai

            converted_msgs = convert_messages_format(messages)

            # # save converted messages to json
            # with open("data/wvs/msgs_1.json", "w") as f:
            #     json.dump(converted_msgs, f)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"],
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)
            if stream:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        reasoning={"effort": reasoning_effort},
                        stream=True,
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        temperature=temperature,
                        stream=True,
                    )
                else:
                    response_stream = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        stream=True,
                    )

                def streamed_chunks():
                    for event in response_stream:
                        if event.type == "response.output_text.delta":
                            yield event.delta

                answer = streamed_chunks()
            else:
                if GPT_SUPPORTS_REASONING:
                    reasoning_effort = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        reasoning={"effort": reasoning_effort},
                    )
                elif GPT_SUPPORTS_TEMPERATURE:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                        temperature=temperature,
                    )
                else:
                    response = client.responses.create(
                        model=GPT_CHAT_MODEL,
                        input=messages,
                    )

                answer = response.output_text
        message = {"role": "assistant", "content": answer}

        # Add the returned value to the messages.
        self.messages_to_display.append(message)

    def display_content(self, content):
        """
        Displays the content of a message in streamlit. Handles plots, strings, and StreamingMessages.
        """
        if isinstance(content, str):
            st.write(content)

        # Visual
        elif isinstance(content, Visual):
            content.show()

        else:
            # So we do this in case
            try:
                content.show()
            except:
                try:
                    st.write(content.get_string())
                except:
                    raise ValueError(
                        f"Message content of type {type(content)} not supported."
                    )

    def display_messages(self):
        """
        Displays visible messages in streamlit. Messages are grouped by role.
        If message content is a Visual, it is displayed in a st.columns((1, 2, 1))[1].
        If the message is a list of strings/Visuals of length n, they are displayed in n columns.
        If a message is a generator, it is displayed with st.write_stream
        Special case: If there are N Visuals in one message, followed by N messages/StreamingMessages in the next, they are paired up into the same N columns.
        """
        # Group by role so user name and avatar is only displayed once

        # st.write(self.messages_to_display)

        for key, group in groupby(self.messages_to_display, lambda x: x["role"]):
            group = list(group)

            if key == "assistant":
                avatar = "data/ressources/img/twelve_chat_logo.svg"
            else:
                try:
                    avatar = st.session_state.user_info["picture"]
                except:
                    avatar = None

            message_block = st.chat_message(name=key, avatar=avatar)
            with message_block:
                for message in group:
                    content = message["content"]
                    if isinstance(content, GeneratorType):
                        final_text = st.write_stream(content)
                        message["content"] = final_text
                    else:
                        self.display_content(content)

    def save_state(self):
        """
        Saves the conversation to session state.
        """
        st.session_state.messages_to_display = self.messages_to_display
        st.session_state.chat_state = self.state


class PlayerChat(Chat):
    def __init__(self, chat_state_hash, player, players, state="empty"):
        self.embeddings = PlayerEmbeddings()
        self.player = player
        self.players = players
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.player.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {
                "role": "system",
                "content": (
                    "You are a UK-based football scout. "
                    f"You ONLY answer questions about {self.player.name} using the statistical data provided to you. "
                    "If the user asks about another player, politely decline and tell them to select that player from the sidebar. "
                    "If the user asks about anything unrelated to football scouting or player statistics, "
                    "politely decline and refocus the conversation on the selected player."
                ),
            },
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of a football scouting platform. "
                    f"The user has selected the player {self.player.name}, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query to provide 2 sentence answers. "
                    "Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the player in terms of data: \n\n"
        description = PlayerDescription(self.player)
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about a player's statistics and what they mean for how they play football."
        ret_val += "The user can select the player they are interested in using the menu to the left."

        return ret_val


class WVSChat(Chat):
    def __init__(
        self,
        chat_state_hash,
        country,
        countries,
        description_dict,
        thresholds_dict,
        state="empty",
    ):
        # TODO:
        self.embeddings = CountryEmbeddings()
        self.country = country
        self.countries = countries
        self.description_dict = description_dict
        self.thresholds_dict = thresholds_dict
        super().__init__(chat_state_hash, state=state)

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.country.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)

    def instruction_messages(self):
        first_messages = [
            {
                "role": "system",
                "content": (
                    "You are a researcher specialising in the World Values Survey. "
                    f"You ONLY answer questions about {self.country.name} using the WVS data provided to you. "
                    "If the user asks about another country, politely decline and tell them to select that country from the sidebar. "
                    "If the user asks about anything unrelated to the World Values Survey or social values research, "
                    "politely decline and refocus the conversation on the selected country's data."
                ),
            },
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of a data analysis platform. "
                    f"The user has selected the country {self.country.name}, and the conversation will be about different core values measured in the World Values Survey study. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query to provide 2 sentence answers. "
                    "Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the country in terms of data: \n\n"
        description = CountryDescription(
            self.country, self.description_dict, self.thresholds_dict
        )
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevant to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about a country's core values."
        ret_val += "The user can select the country they are interested in using the menu to the left."

        return ret_val


class PersonChat(Chat):
    def __init__(self, chat_state_hash, person, persons, state="empty"):
        self.embeddings = PersonEmbeddings()
        self.person = person
        self.persons = persons
        super().__init__(chat_state_hash, state=state)

    def instruction_messages(self):
        first_messages = [
            {
                "role": "system",
                "content": (
                    "You are a recruiter analysing personality test results. "
                    f"You ONLY answer questions about {self.person.name} using the personality data provided to you. "
                    "If the user asks about another person, politely decline and tell them to select that person from the sidebar. "
                    "If the user asks about anything unrelated to personality assessment or the data available, "
                    "politely decline and refocus the conversation on the selected person."
                ),
            },
            {
                "role": "user",
                "content": (
                    "After these messages you will be interacting with a user of a personality test platform. "
                    f"The user has selected the person {self.person.name}, and the conversation will be about them. "
                    "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                    "All user messages will be prefixed with 'User:' and enclosed with ```. "
                    "When responding to the user, speak directly to them. "
                    "Use the information provided before the query to provide 2 sentence answers. "
                    "Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                ),
            },
        ]
        return first_messages

    def get_relevant_info(self, query):

        # If there is no query then use the last message from the user
        if query == "":
            query = self.visible_messages[-1]["content"]

        ret_val = "Here is a description of the person in terms of data: \n\n"
        description = PersonDescription(self.person)
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"
        ret_val += "\n".join(results["assistant"].to_list())

        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about person's statistics and what they mean about their personality."
        ret_val += "The user can select the persons they are interested in using the menu to the left."

        return ret_val

    def get_input(self):
        """
        Get input from streamlit."""

        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.person.name}?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )

            self.handle_input(x, stream=True)


class PressingChat(Chat):
    """
    Chat agent for the pressing analyst page.
    Uses four tools via function calling (tool_choice="auto") and a two-step LLM workflow.
    """

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

    def __init__(self, chat_state_hash, team, pressing_stats, state="empty"):
        self.team = team
        self.pressing_stats = pressing_stats
        self.embeddings = PressingEmbeddings()
        super().__init__(chat_state_hash, state=state)

    def instruction_messages(self):
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
                    f"The user has selected {self.team.name}. The conversation focuses on their pressing. "
                    "You have four tools: get_team_pressing_summary, search_pressing_knowledge, "
                    "compare_team_pressing, and get_league_rankings. "
                    "Always use a tool to ground your answer in data. "
                    "Give concise 2-3 sentence responses directly to the user."
                ),
            },
        ]

    # ── Tool implementations ───────────────────────────────────────────────────

    def _get_team_pressing_summary(self):
        description = PressingDescription(self.team)
        return description.synthesize_text()

    def _search_pressing_knowledge(self, query):
        results = self.embeddings.search(query, top_n=3)
        if results.empty:
            return "No relevant pressing knowledge found for that query."
        return "\n\n".join(results["assistant"].tolist())

    def _compare_team_pressing(self, other_team_name):
        from classes.data_source import PressingStats
        df = self.pressing_stats.df
        match = df[df["team_name"].str.lower().str.contains(other_team_name.lower(), na=False)]
        if match.empty:
            return f"Could not find a team matching '{other_team_name}' in the league data."
        other = match.iloc[0]
        my = df[df["team_id"] == self.team.id].iloc[0]
        n = len(df)
        lines = [f"Pressing comparison: {self.team.name} vs {other['team_name']} ({n} Premier League teams)\n"]
        for metric in PressingStats.METRICS:
            phrase = PressingStats.METRIC_PHRASES[metric]
            my_val = my[metric]
            other_val = other[metric]
            is_negative = metric in PressingStats.NEGATIVE_METRICS

            rank_col = f"{metric}_Ranks"
            if rank_col in my.index:
                my_rank_raw = int(my[rank_col])
                other_rank_raw = int(other[rank_col])
                # Ranks are stored as rank-by-value (1 = highest). For negative metrics
                # a lower value is better, so invert to a "performance rank" where 1 = best.
                if is_negative:
                    my_rank = n + 1 - my_rank_raw
                    other_rank = n + 1 - other_rank_raw
                else:
                    my_rank = my_rank_raw
                    other_rank = other_rank_raw
                rank_info = f"{self.team.name} {my_rank}/{n}, {other['team_name']} {other_rank}/{n}"
            else:
                rank_info = ""

            if is_negative:
                leader = self.team.name if my_val < other_val else (other["team_name"] if other_val < my_val else None)
            else:
                leader = self.team.name if my_val > other_val else (other["team_name"] if other_val > my_val else None)
            edge = f"{leader} leads" if leader else "equal"

            line = f"  {phrase}: {self.team.name} {my_val*100:.1f}% vs {other['team_name']} {other_val*100:.1f}%"
            if rank_info:
                line += f" (league rank — {rank_info})"
            line += f" — {edge}"
            lines.append(line)
        return "\n".join(lines)

    def _get_league_rankings(self):
        from classes.data_source import PressingStats
        df = self.pressing_stats.df.copy()
        n = len(df)
        lines = [f"League rankings for {self.team.name} (out of {n} Premier League teams):\n"]
        for metric in PressingStats.METRICS:
            phrase = PressingStats.METRIC_PHRASES[metric]
            ascending = metric in PressingStats.NEGATIVE_METRICS
            rank_series = df[metric].rank(ascending=ascending, method="min")
            rank = int(rank_series[df["team_id"] == self.team.id].values[0])
            lines.append(f"  {phrase}: ranked {rank} of {n}")
        return "\n".join(lines)

    def _execute_tool(self, name, args):
        if name == "get_team_pressing_summary":
            return self._get_team_pressing_summary()
        if name == "search_pressing_knowledge":
            return self._search_pressing_knowledge(args.get("query", ""))
        if name == "compare_team_pressing":
            return self._compare_team_pressing(args.get("other_team_name", ""))
        if name == "get_league_rankings":
            return self._get_league_rankings()
        return "Unknown tool requested."

    # ── Two-step handle_input ──────────────────────────────────────────────────

    def handle_input(self, input, reasoning_effort=None, temperature=1, stream=False):
        messages = self.instruction_messages()
        for m in self.messages_to_display:
            if isinstance(m["content"], str):
                messages.append({"role": m["role"], "content": m["content"]})

        self.messages_to_display.append({"role": "user", "content": input})
        messages.append({"role": "user", "content": input})

        st.expander("Chat transcript", expanded=False).write(messages)

        client = OpenAI(api_key=GPT_KEY, base_url=GPT_BASE)

        # ── Step 1: let the model pick a tool ─────────────────────────────────
        kwargs1 = {
            "model": GPT_CHAT_MODEL,
            "input": messages,
            "tools": self.TOOLS,
            "tool_choice": "auto",
        }
        if GPT_SUPPORTS_REASONING:
            re = reasoning_effort if reasoning_effort in GPT_AVAILABLE_REASONING_EFFORTS else GPT_AVAILABLE_REASONING_EFFORTS[0]
            kwargs1["reasoning"] = {"effort": re}
        elif GPT_SUPPORTS_TEMPERATURE:
            kwargs1["temperature"] = temperature

        response1 = client.responses.create(**kwargs1)

        tool_calls = [item for item in response1.output if getattr(item, "type", None) == "function_call"]

        # ── Step 2: execute tool and generate final answer ─────────────────────
        if tool_calls:
            tool_call = tool_calls[0]
            args = json.loads(tool_call.arguments) if tool_call.arguments else {}
            tool_result = self._execute_tool(tool_call.name, args)

            kwargs2 = {
                "model": GPT_CHAT_MODEL,
                "previous_response_id": response1.id,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": tool_result,
                    }
                ],
            }
            if GPT_SUPPORTS_REASONING:
                kwargs2["reasoning"] = {"effort": re}
            elif GPT_SUPPORTS_TEMPERATURE:
                kwargs2["temperature"] = temperature

            response2 = client.responses.create(**kwargs2)
            answer = response2.output_text
        else:
            answer = response1.output_text

        self.messages_to_display.append({"role": "assistant", "content": answer})

    def get_input(self):
        if x := st.chat_input(
            placeholder=f"What else would you like to know about {self.team.name}'s pressing?"
        ):
            if len(x) > 500:
                st.error(
                    f"Your message is too long ({len(x)} characters). Please keep it under 500 characters."
                )
            self.handle_input(x)
