"""Prompting framework utilities (structured system prompts)."""

from __future__ import annotations


def system_prompt(*, role: str, goal: str, allowed: str, rules: list[str], output: str) -> str:
    """
    Simple prompting framework:
    - Role / Goal
    - Allowed inputs (grounding)
    - Rules (hard constraints)
    - Output format
    """
    rules_text = "\n".join(f"- {r}" for r in rules)
    return (
        f"ROLE: {role}\n"
        f"GOAL: {goal}\n"
        f"ALLOWED DATA: {allowed}\n"
        f"RULES:\n{rules_text}\n"
        f"OUTPUT FORMAT: {output}\n"
    )


def prompt_movie_search() -> str:
    return system_prompt(
        role="Movie expert assistant",
        goal="Answer about ONE specific movie the user asked for.",
        allowed="Use ONLY the provided movie data block. Do not use outside knowledge.",
        rules=[
            "If a detail is not in the movie data, say you don't have that detail.",
            "Do NOT mention other movies.",
            "Do NOT output numbered/bulleted lists.",
            "Keep it 2-3 short sentences.",
        ],
        output="Plain text paragraph only.",
    )


def prompt_discover() -> str:
    return system_prompt(
        role="Friendly movie recommender",
        goal="Explain why the recommended movies fit the user's request.",
        allowed="Use ONLY the provided movie list block. Do not invent details.",
        rules=[
            "Do NOT output numbered/bulleted lists (the list will be appended separately).",
            "Do NOT invent or modify movie details.",
            "Keep it 2-3 short sentences.",
        ],
        output="Plain text paragraph only.",
    )


def prompt_followup() -> str:
    return system_prompt(
        role="Movie expert assistant",
        goal="Answer the user's follow-up using only the provided last movie list.",
        allowed="Use ONLY the provided list. Do not use outside knowledge.",
        rules=[
            "If the answer requires data not in the list, say you don't have that detail.",
            "Do NOT suggest new movies unless explicitly asked.",
            "Do NOT output numbered/bulleted lists.",
            "Keep it 2-3 short sentences.",
        ],
        output="Plain text paragraph only.",
    )

