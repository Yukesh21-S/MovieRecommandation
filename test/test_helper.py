from movie_chatbot.agent.helpers import (
    extract_title,
    fuzzy_match_title,
    sanitize_commentary,
    is_followup,
    parse_movie_block,
    extract_requested_index,
)

# ── extract_title ──────────────────────────────────────────
def test_extract_title_removes_find_prefix():
    assert extract_title("find Inception") == "Inception"

def test_extract_title_removes_tell_me_about():
    assert extract_title("tell me about Parasite") == "Parasite"

def test_extract_title_removes_movie_suffix():
    assert extract_title("Interstellar movie") == "Interstellar"

def test_extract_title_no_prefix_unchanged():
    assert extract_title("Inception") == "Inception"

# ── fuzzy_match_title ──────────────────────────────────────
def test_fuzzy_match_finds_close_match():
    titles = ["Inception", "Interstellar", "The Dark Knight"]
    assert fuzzy_match_title("inceptoin", titles) == "Inception"

def test_fuzzy_match_returns_none_for_poor_match():
    titles = ["Inception", "Interstellar"]
    assert fuzzy_match_title("xyz123", titles) is None

def test_fuzzy_match_empty_query_returns_none():
    assert fuzzy_match_title("", ["Inception"]) is None

# ── sanitize_commentary ────────────────────────────────────
def test_sanitize_removes_numbered_list():
    text = "Great movies!\n1. Inception (2010)"
    assert sanitize_commentary(text) == "Great movies!"

def test_sanitize_keeps_plain_commentary():
    text = "Here are some great picks for you."
    assert sanitize_commentary(text) == text

# ── is_followup ────────────────────────────────────────────
def test_is_followup_detects_hint_keyword():
    history = [{"role": "assistant", "content": "Here are some movies..."}]
    assert is_followup("tell me more about the first one", history) is True

def test_is_followup_false_with_no_history():
    assert is_followup("recommend a horror movie", []) is False

def test_is_followup_false_for_fresh_query():
    history = [{"role": "assistant", "content": "Here are some movies..."}]
    assert is_followup("suggest me a tamil thriller", history) is False

# ── extract_requested_index ────────────────────────────────
def test_extract_index_from_ordinal_word():
    assert extract_requested_index("tell me about the third one") == 3

def test_extract_index_from_number():
    assert extract_requested_index("what about movie 2") == 2

def test_extract_index_none_for_no_number():
    assert extract_requested_index("recommend something fun") is None

# ── parse_movie_block ──────────────────────────────────────
def test_parse_movie_block_returns_correct_fields():
    block = (
        "1. Inception (2010) | Action, Sci-Fi | ⭐ 8.8/10\n"
        "   A thief who steals corporate secrets."
    )
    items = parse_movie_block(block)
    assert len(items) == 1
    assert items[0]["title"] == "Inception"
    assert items[0]["year"] == "2010"
    assert float(items[0]["rating"]) == 8.8