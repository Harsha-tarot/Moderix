# Content Moderation OpenEnv — Implementation Start Here

**This file contains ready-to-use code scaffolds. Copy-paste, then adapt.**

---

## **STEP 1: Project Structure** (5 minutes)

```bash
mkdir -p content-moderation-openenv
cd content-moderation-openenv

# Create directories
mkdir -p data graders tests

# Initialize git
git init
git config user.email "you@example.com"
git config user.name "Your Name"

# Create venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base dependencies
pip install pydantic aiohttp openai python-dotenv
pip freeze > requirements.txt
```

---

## **STEP 2: Create Core Environment File (my_env.py)**

Copy this scaffold and fill in the `###` sections:

```python
# my_env.py
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
import json

class Observation(BaseModel):
    """What the agent observes at each step."""
    content_id: str
    content_text: str
    source: str  # "twitter", "reddit", "discord"
    timestamp: str
    metadata: dict = Field(default_factory=dict)

class Action(BaseModel):
    """What the agent decides to do."""
    decision: str  # "approve", "review", "reject", "escalate"
    reasoning: str  # Why this decision
    confidence: float = 0.5  # 0.0-1.0

class ContentModerationEnv:
    """OpenEnv-compliant content moderation environment."""
    
    def __init__(self, task_name: str = "content_moderation"):
        self.task_name = task_name
        self.current_step = 0
        self.max_steps = 8
        self.episode_rewards = []
        self.decisions_made = []
        self.gold_labels = {}
        self.current_batch = []
        self.batch_index = 0
        
    async def initialize(self):
        """Load gold labels and initialize."""
        ### STEP 1: Load training_set.json from data/
        try:
            with open("data/training_set.json", "r") as f:
                dataset = json.load(f)
                self.gold_labels = {item["id"]: item for item in dataset}
                print(f"[INFO] Loaded {len(self.gold_labels)} gold labels")
        except FileNotFoundError:
            print("[WARN] training_set.json not found, using empty labels")
            self.gold_labels = {}
    
    async def reset(self) -> Observation:
        """Initialize a new episode."""
        self.current_step = 0
        self.episode_rewards = []
        self.decisions_made = []
        
        ### STEP 2: Load a batch of posts (8 posts from gold_labels)
        posts = list(self.gold_labels.values())[:8]
        self.current_batch = posts
        self.batch_index = 0
        
        if not self.current_batch:
            # Fallback: create a dummy observation
            print("[WARN] No posts in batch, returning dummy observation")
            return Observation(
                content_id="dummy_001",
                content_text="[No content available]",
                source="test",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Return first post as observation
        post = self.current_batch[self.batch_index]
        return Observation(
            content_id=post["id"],
            content_text=post["content"],
            source=post.get("source", "twitter"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"batch_size": len(self.current_batch)}
        )
    
    async def step(self, action: Action) -> tuple:
        """Process one moderation decision."""
        self.current_step += 1
        
        # Get gold label for this post
        current_post = self.current_batch[self.batch_index] if self.batch_index < len(self.current_batch) else None
        
        if not current_post:
            return Observation(
                content_id="end",
                content_text="[Episode ended]",
                source="test",
                timestamp=datetime.now(timezone.utc).isoformat()
            ), 0.0, True, {"error": "No more posts"}
        
        ### STEP 3: Grade this decision against gold labels
        reward = self._grade_decision(action, current_post)
        self.episode_rewards.append(reward)
        self.decisions_made.append(action.decision)
        
        # Move to next post
        self.batch_index += 1
        done = (self.batch_index >= len(self.current_batch)) or (self.current_step >= self.max_steps)
        
        # Return next observation or end signal
        if not done and self.batch_index < len(self.current_batch):
            next_post = self.current_batch[self.batch_index]
            obs = Observation(
                content_id=next_post["id"],
                content_text=next_post["content"],
                source=next_post.get("source", "twitter"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"step": self.current_step, "cumulative_reward": sum(self.episode_rewards)}
            )
        else:
            obs = Observation(
                content_id="done",
                content_text="[Episode complete]",
                source="test",
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"final_reward": sum(self.episode_rewards)}
            )
        
        info = {
            "step": self.current_step,
            "cumulative_reward": sum(self.episode_rewards),
            "decision": action.decision,
            "reasoning": action.reasoning[:50]  # truncate for logging
        }
        
        return obs, reward, done, info
    
    def _grade_decision(self, action: Action, post: dict) -> float:
        """Grade the agent's decision against gold labels."""
        ### STEP 4: Implement grading logic
        
        decision = action.decision.lower()
        confidence = action.confidence
        
        # Simple accuracy-based grading
        # Toxicity task
        if "toxicity" in post:
            gold_tox = post["toxicity"]
            if gold_tox > 0.5:  # Toxic post
                if decision in ["review", "reject"]:
                    reward = 0.8 + 0.2 * min(confidence, 0.5)
                else:
                    reward = 0.1
            else:  # Safe post
                if decision == "approve":
                    reward = 0.9 + 0.1 * confidence
                else:
                    reward = 0.3
        else:
            reward = 0.5  # Neutral if no gold label
        
        return min(max(reward, 0.0), 1.0)  # Clamp to [0, 1]
    
    async def state(self) -> dict:
        """Return current episode state."""
        return {
            "step": self.current_step,
            "max_steps": self.max_steps,
            "episode_rewards": self.episode_rewards,
            "decisions_made": self.decisions_made,
            "cumulative_reward": sum(self.episode_rewards),
            "batch_size": len(self.current_batch)
        }
    
    async def close(self):
        """Cleanup (if needed)."""
        pass
    
    @classmethod
    async def from_docker_image(cls, image_name: str):
        """For HF Space compatibility."""
        env = cls()
        await env.initialize()
        return env
    
    @classmethod
    async def from_env(cls, **kwargs):
        """Initialize from environment variables."""
        env = cls(**kwargs)
        await env.initialize()
        return env

# Test locally
if __name__ == "__main__":
    async def test():
        env = await ContentModerationEnv.from_env()
        obs = await env.reset()
        print(f"Reset: {obs.content_id}, {obs.content_text[:50]}")
        
        # Simulate one step
        action = Action(decision="approve", reasoning="Safe post", confidence=0.8)
        obs, reward, done, info = await env.step(action)
        print(f"Step 1: reward={reward}, done={done}")
        
        await env.close()
    
    asyncio.run(test())
```

---

## **STEP 3: Create Training Dataset (data/training_set.json)**

Use this template and add 50 real posts:

```json
[
  {
    "id": "post_001",
    "content": "Just finished my morning jog. Feeling great!",
    "source": "twitter",
    "toxicity": 0.0,
    "spam": false,
    "nsfw_category": "safe",
    "justification": "Innocent personal update, no issues"
  },
  {
    "id": "post_002",
    "content": "CLICK HERE TO WIN $$$$ FREE MONEY NOW!!!",
    "source": "reddit",
    "toxicity": 0.1,
    "spam": true,
    "nsfw_category": "safe",
    "justification": "Classic spam pattern: urgency, caps, unusual punctuation"
  },
  {
    "id": "post_003",
    "content": "I hate all [group]. They should be [violent action].",
    "source": "discord",
    "toxicity": 0.95,
    "spam": false,
    "nsfw_category": "safe",
    "justification": "Hate speech and calls for violence, high toxicity"
  },
  {
    "id": "post_004",
    "content": "Check out my NSFW content at [link] OnlyFans",
    "source": "twitter",
    "toxicity": 0.2,
    "spam": true,
    "nsfw_category": "explicit",
    "justification": "Spam + NSFW promotion"
  },
  {
    "id": "post_005",
    "content": "News: Breaking violence in downtown area, 3 injured",
    "source": "reddit",
    "toxicity": 0.3,
    "spam": false,
    "nsfw_category": "violence",
    "justification": "Violence content, but journalistic context (lower toxicity)"
  }
]
```

**Action:** Fill this with 50 diverse posts. For speed, use:
- 10 safe posts (toxicity 0.0, spam false, nsfw safe)
- 10 toxic posts (toxicity 0.6-0.95, spam false, nsfw safe)
- 10 spam posts (toxicity 0.1-0.3, spam true, various nsfw)
- 10 NSFW posts (toxicity variable, spam false, nsfw 4 categories)
- 10 borderline/tricky (challenge the agent)

---

## **STEP 4: Create Graders (graders/*.py)**

### **graders/toxicity_grader.py**

```python
def grade_toxicity(content: str, predicted_score: float, gold_label_score: float) -> float:
    """
    Grade toxicity detection.
    
    Args:
        content: Post text (for reference)
        predicted_score: Agent's toxicity score (0.0-1.0)
        gold_label_score: True label (0.0-1.0)
    
    Returns:
        Reward (0.0-1.0)
    """
    # Sigmoid accuracy: penalize distance from gold label
    distance = abs(predicted_score - gold_label_score)
    
    # Linear penalty up to distance=1.0
    accuracy = max(0.0, 1.0 - distance)
    
    # Confidence bonus: if agent is confident and correct
    confidence_bonus = 0.0
    if gold_label_score > 0.5 and predicted_score > 0.5:
        # Both say toxic
        confidence_bonus = 0.1 * min(predicted_score, gold_label_score)
    elif gold_label_score < 0.5 and predicted_score < 0.5:
        # Both say safe
        confidence_bonus = 0.1 * (1 - max(predicted_score, gold_label_score))
    
    reward = accuracy + confidence_bonus * 0.1
    return min(max(reward, 0.0), 1.0)


# Test
if __name__ == "__main__":
    # Test case 1: Perfect prediction
    r1 = grade_toxicity("bad content", 0.9, 0.9)
    print(f"Perfect: {r1}")  # Should be ~1.0
    
    # Test case 2: Off by 0.2
    r2 = grade_toxicity("bad content", 0.7, 0.9)
    print(f"Off by 0.2: {r2}")  # Should be ~0.8
    
    # Test case 3: Completely wrong
    r3 = grade_toxicity("bad content", 0.1, 0.9)
    print(f"Wrong: {r3}")  # Should be ~0.0
```

### **graders/spam_grader.py**

```python
def grade_spam(content: str, predicted_is_spam: bool, gold_is_spam: bool) -> float:
    """
    Grade spam classification (binary).
    
    Args:
        content: Post text (for reference)
        predicted_is_spam: Agent's binary prediction
        gold_is_spam: True label
    
    Returns:
        Reward (0.0-1.0)
    """
    # Basic accuracy
    if predicted_is_spam == gold_is_spam:
        # Correct prediction
        if gold_is_spam:
            # Caught spam (important, recall matters)
            reward = 1.0
        else:
            # Correctly approved safe post
            reward = 0.9
    else:
        # Incorrect prediction
        if gold_is_spam and not predicted_is_spam:
            # Missed spam (worse than false positive)
            reward = 0.2
        else:
            # False positive (less bad)
            reward = 0.5
    
    return min(max(reward, 0.0), 1.0)


if __name__ == "__main__":
    r1 = grade_spam("...", True, True)  # Caught spam
    print(f"Caught spam: {r1}")  # 1.0
    
    r2 = grade_spam("...", False, False)  # Safe post
    print(f"Safe post: {r2}")  # 0.9
    
    r3 = grade_spam("...", False, True)  # Missed spam
    print(f"Missed spam: {r3}")  # 0.2
```

### **graders/nsfw_grader.py**

```python
def grade_nsfw(content: str, predicted_category: str, gold_category: str) -> float:
    """
    Grade NSFW multi-class detection.
    
    Categories: "safe", "violence", "explicit", "adult_content"
    
    Args:
        content: Post text
        predicted_category: Agent's prediction
        gold_category: True label
    
    Returns:
        Reward (0.0-1.0)
    """
    if predicted_category == gold_category:
        # Perfect match
        return 1.0
    
    # Check if "close" (confusable categories)
    close_pairs = {
        ("violence", "explicit"),
        ("adult_content", "explicit"),
        ("safe", "violence"),
        ("safe", "explicit")
    }
    
    if (predicted_category, gold_category) in close_pairs or \
       (gold_category, predicted_category) in close_pairs:
        # One category off (understandable confusion)
        return 0.6
    else:
        # Completely wrong
        return 0.1


if __name__ == "__main__":
    r1 = grade_nsfw("...", "violence", "violence")
    print(f"Perfect: {r1}")  # 1.0
    
    r2 = grade_nsfw("...", "explicit", "violence")
    print(f"One off: {r2}")  # 0.6
    
    r3 = grade_nsfw("...", "safe", "explicit")
    print(f"Wrong: {r3}")  # 0.1
```

---

## **STEP 5: Create inference.py (Baseline Script)**

```python
# inference.py
import asyncio
import os
import json
import sys
from typing import List, Optional
from openai import OpenAI
from my_env import ContentModerationEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
TASK_NAME = "content_moderation"
BENCHMARK = "openenv_moderation"
MAX_STEPS = 8

SYSTEM_PROMPT = """You are a content moderation agent. Your job is to classify and route user-generated content.

For each post, respond with EXACTLY this JSON format:
{
  "decision": "approve" | "review" | "reject" | "escalate",
  "reasoning": "Brief justification (max 100 chars)",
  "confidence": 0.5
}

Decision guide:
- "approve": Safe content, no issues (confidence > 0.8)
- "review": Moderate concern, needs human review (confidence 0.5-0.7)
- "reject": Clear policy violation, high toxicity/NSFW (confidence > 0.8)
- "escalate": Uncertain, ambiguous, needs escalation (confidence < 0.5)

Be decisive but calibrated."""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_short = action[:40] if len(action) > 40 else action
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[END] success={str(success).lower()} steps={steps} avg_reward={avg_reward:.2f} rewards={rewards_str}", flush=True)

def get_model_response(client: OpenAI, content: str, step: int) -> dict:
    """Get moderation decision from LLM."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Step {step}. Moderate this post:\n\n{content}"},
            ],
            temperature=0.3,
            max_tokens=200,
            stream=False,
        )
        response_text = completion.choices[0].message.content or "{}"
        
        # Try to parse JSON
        try:
            parsed = json.loads(response_text)
            return {
                "decision": parsed.get("decision", "escalate"),
                "reasoning": parsed.get("reasoning", "No reasoning provided"),
                "confidence": float(parsed.get("confidence", 0.5))
            }
        except json.JSONDecodeError:
            print(f"[DEBUG] LLM response not JSON: {response_text[:100]}", flush=True)
            return {
                "decision": "escalate",
                "reasoning": "Could not parse LLM response",
                "confidence": 0.3
            }
    except Exception as e:
        print(f"[DEBUG] LLM request failed: {e}", flush=True)
        return {
            "decision": "escalate",
            "reasoning": "API error",
            "confidence": 0.2
        }

async def main() -> None:
    # Initialize
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await ContentModerationEnv.from_env()
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset and get first observation
        obs = await env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            # Get LLM decision
            response = get_model_response(client, obs.content_text, step)
            action = Action(
                decision=response["decision"],
                reasoning=response["reasoning"],
                confidence=response["confidence"]
            )
            
            # Step environment
            obs, reward, done, info = await env.step(action)
            
            rewards.append(reward)
            steps_taken = step
            
            action_str = f"{action.decision}:{action.reasoning[:30]}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            
            if done:
                break
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        success = avg_reward >= 0.5  # Threshold for success
        
    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## **STEP 6: Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY my_env.py .
COPY inference.py .
COPY graders/ ./graders/
COPY data/ ./data/

# Set environment defaults
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV HF_TOKEN=hf_xxxxx

CMD ["python", "inference.py"]
```

---

## **STEP 7: Create openenv.yaml**

```yaml
name: content-moderation
version: 1.0.0
description: "Autonomous content moderation agent that classifies, flags, and routes social media posts across toxicity, spam, and NSFW categories."
author: "Your Name"

observation:
  type: object
  required: [content_id, content_text, source, timestamp]
  properties:
    content_id:
      type: string
    content_text:
      type: string
      description: "Full text of post to moderate"
    source:
      type: string
      enum: ["twitter", "reddit", "discord"]
    timestamp:
      type: string
      format: iso8601
    metadata:
      type: object

action:
  type: object
  required: [decision, reasoning]
  properties:
    decision:
      type: string
      enum: ["approve", "review", "reject", "escalate"]
    reasoning:
      type: string
    confidence:
      type: number
      minimum: 0.0
      maximum: 1.0

reward:
  type: number
  minimum: 0.0
  maximum: 1.0

tasks:
  - id: toxicity_detection
    name: Toxicity Detection
    description: Classify posts for toxic language severity
    difficulty: easy
    
  - id: spam_classification
    name: Spam Classification
    description: Binary spam vs legitimate
    difficulty: medium
    
  - id: nsfw_detection
    name: NSFW Detection
    description: Categorize inappropriate content
    difficulty: hard
    
  - id: reasoning_quality
    name: Reasoning Quality
    description: Justification quality
    difficulty: medium
```

---

## **NEXT STEPS**

1. Copy all code above into your repo
2. Fill in 50 realistic posts in `data/training_set.json`
3. Test locally: `python inference.py`
4. Fix any errors
5. Push to GitHub
6. Deploy to HF Space
7. **SUBMIT** 72 hours before deadline

**You're ready to build.** Good luck! 🚀
