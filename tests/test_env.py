import asyncio

import pytest
import logging
logging.basicConfig(level=logging.DEBUG)

from my_env import Action, ContentModerationEnv


@pytest.fixture
def env():
    e = ContentModerationEnv()
    asyncio.run(e.initialize())
    return e


@pytest.mark.asyncio
async def test_reward_bounds(env):
    """Test that rewards are strictly between 0.0 and 1.0"""
    await env.reset()

    # Action 1: Perfect approve
    action = Action(
        decision="approve",
        violation_type="none",
        reasoning="Looks fine",
        confidence=1.0,
    )
    _, reward, _, _ = await env.step(action)
    assert 0.0 <= reward.value <= 1.0

    # Action 2: Catastrophic false ban
    env.current_batch[env.batch_index] = {
        "id": "test_benign",
        "content": "Hello world",
        "toxicity": 0.0,
        "spam": False,
        "nsfw_category": "safe",
        "justification": "Safe",
    }
    action = Action(
        decision="ban_user", violation_type="toxicity", reasoning="Die", confidence=1.0
    )
    _, reward, _, _ = await env.step(action)
    assert reward.value == 0.0  # False ban should strictly be 0.0


@pytest.mark.asyncio
async def test_user_reputation(env):
    """Test that user reputation correctly increments/decrements"""
    obs = await env.reset()
    initial_rep = obs.user_reputation

    # Force the environment to evaluate a benign post
    env.current_batch[env.batch_index] = {
        "id": "test1",
        "content": "Hello world",
        "toxicity": 0.0,
        "spam": False,
        "nsfw_category": "safe",
        "justification": "Safe",
    }
    action = Action(
        decision="approve", violation_type="none", reasoning="Safe", confidence=1.0
    )
    await env.step(action)

    # Reputation should go up (max 1.0)
    assert env.user_reputation == min(1.0, initial_rep + 0.1)

    # Force a malicious post
    env.current_batch[env.batch_index] = {
        "id": "test2",
        "content": "I will kill you",
        "toxicity": 0.95,
        "spam": False,
        "nsfw_category": "safe",
        "justification": "Threat",
    }
    action = Action(
        decision="approve", violation_type="none", reasoning="Safe", confidence=1.0
    )
    await env.step(action)

    # Reputation should go down for failing to catch it
    assert env.user_reputation < 1.0
