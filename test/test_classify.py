from movie_chatbot.agent.constants import LITE_MOOD_KEYWORDS, LATEST_KEYWORDS

def test_lite_mood_keywords_present():
    # stress and tired should trigger lite mood
    assert "stress" in LITE_MOOD_KEYWORDS
    assert "tired" in LITE_MOOD_KEYWORDS

def test_latest_keywords_present():
    assert "latest" in LATEST_KEYWORDS
    assert "recent" in LATEST_KEYWORDS

def test_lite_mood_detection_from_query():
    query = "I am stressed recommend something fun"
    words = set(query.lower().split())
    assert bool(LITE_MOOD_KEYWORDS & words) is True

def test_latest_detection_from_query():
    query = "show me the latest horror movies"
    words = set(query.lower().split())
    assert bool(LATEST_KEYWORDS & words) is True