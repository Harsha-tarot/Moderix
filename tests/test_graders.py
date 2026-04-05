import pytest

from graders.nsfw_grader import grade_nsfw
from graders.reasoning_grader import grade_reasoning
from graders.spam_grader import grade_spam
from graders.toxicity_grader import grade_toxicity


def test_toxicity_grader():
    # Perfect match
    assert grade_toxicity("content", 0.9, 0.9) > 0.9
    # Complete miss
    assert grade_toxicity("content", 0.1, 0.9) < 0.2


def test_spam_grader():
    assert grade_spam("content", True, True) == 1.0
    assert grade_spam("content", False, False) == 0.9
    assert grade_spam("content", False, True) == 0.2
    assert grade_spam("content", True, False) == 0.5


def test_nsfw_grader():
    assert grade_nsfw("content", "explicit", "explicit") == 1.0
    assert grade_nsfw("content", "explicit", "violence") == 0.6  # Confusable pair
    assert grade_nsfw("content", "safe", "explicit") == 0.6
    assert grade_nsfw("content", "safe", "adult_content") == 0.1


def test_reasoning_grader():
    r1 = grade_reasoning("Contains explicitly toxic threats", "Explicit toxic threat")
    r2 = grade_reasoning(
        "A completely safe and wholesome post", "Explicit toxic threat"
    )
    assert r1 > r2
    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r2 <= 1.0
