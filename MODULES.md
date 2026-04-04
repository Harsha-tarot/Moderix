# MODULES.md: Internal Module Reference

This document provides detailed reference for the 6 core modules in the content moderation OpenEnv, intended for developers extending or modifying the environment.

## Module 1: ContentPool

The ContentPool class manages loading and serving content batches from a JSON dataset. It stores 50 posts with gold labels and provides methods to retrieve batches for episodes. Key features include schema validation and optional shuffling. Configuration includes data_path (default: data/training_set.json) and batch_size (default: 8).

Public methods:
- `load_from_json(path)`: Loads 50 posts from JSON file
- `get_batch(size=8)`: Returns next batch of posts
- `rotate_batch()`: Move to next batch
- `get_by_id(post_id)`: Retrieve specific post

Private methods:
- `_validate_schema()`: Ensure JSON has gold labels
- `_shuffle()`: Optional randomization

Example:
```python
pool = ContentPool(data_path="data/training_set.json")
batch = pool.get_batch(size=8)
```

## Module 2: ClassificationEngine

The ClassificationEngine class handles classification of posts across three dimensions: toxicity, spam, and NSFW. It uses keyword scanning, pattern matching, and heuristics to produce scores and categories. Tuning parameters include keyword lists and thresholds for caps ratio and URL suspicion.

Public methods:
- `classify_toxicity(text) → float (0.0–1.0)`: Keyword matching + heuristics
- `classify_spam(text) → bool`: Pattern matching (URL, caps, keywords)
- `classify_nsfw(text) → str`: Multi-class classification ("safe" | "violence" | "explicit" | "adult_content")

Private methods:
- `_extract_keywords(text)`: Set of detected keywords
- `_count_caps(text)`: Caps ratio
- `_detect_urls(text)`: URL patterns found

Configuration:
- `toxicity_keywords`: ["hate", "kill", "violence", ...]
- `spam_keywords`: ["free", "click", "buy", ...]
- `nsfw_keywords_violence`: ["blood", "gore", ...]
- `nsfw_keywords_explicit`: ["sex", "porn", ...]
- `caps_threshold`: 0.5
- `url_suspicion_threshold`: 0.7

Example:
```python
engine = ClassificationEngine()
tox_score = engine.classify_toxicity("I hate [group]")  # → 0.92
is_spam = engine.classify_spam("CLICK HERE FOR $$$")    # → True
nsfw_cat = engine.classify_nsfw("Violence post...")     # → "violence"
```

## Module 3: RoutingEngine

The RoutingEngine class converts classification results into agent decisions using if-then rules based on risk scores. It applies thresholds for toxicity, spam confidence, and NSFW categories to determine approve/review/reject/escalate.

Public methods:
- `route(toxicity, spam, nsfw) → str`: Returns "approve" | "review" | "reject" | "escalate"

Decision logic:
- If toxicity > 0.75: "reject"
- If spam=True and confidence high: "reject"
- If NSFW != "safe": "review" or "reject"
- If all < thresholds: "approve"
- If uncertain: "escalate"

Configuration:
- `toxicity_threshold_reject`: 0.75
- `toxicity_threshold_review`: 0.4
- `spam_confidence_threshold`: 0.8
- `nsfw_threshold_review`: 0.6

Example:
```python
router = RoutingEngine()
decision = router.route(toxicity=0.85, spam=False, nsfw="safe")  # → "reject"
```

## Module 4: GraderModule

The GraderModule class evaluates agent decisions against gold labels, computing rewards for each task. It uses distance metrics for toxicity, F1 scores for spam/NSFW, and semantic similarity for reasoning.

Public methods:
- `grade_toxicity(pred_score, gold_score) → float`: Reward based on distance
- `grade_spam(pred_is_spam, gold_is_spam) → float`: F1-weighted, recall emphasis
- `grade_nsfw(pred_category, gold_category) → float`: Macro F1 across 4 categories
- `grade_reasoning(agent_reasoning, reference_reasoning) → float`: Semantic similarity

Dependencies: sentence-transformers, numpy

Configuration:
- `similarity_model`: "sentence-transformers/all-MiniLM-L6-v2"
- `min_reasoning_length`: 10
- `max_reasoning_length`: 500

Example:
```python
grader = GraderModule()
reward = grader.grade_toxicity(pred=0.8, gold=0.85)  # → 0.97
```

## Module 5: RewardShaper

The RewardShaper class combines task rewards into a single episode reward, with optional confidence bonuses.

Public methods:
- `compute_reward(toxicity_r, spam_r, nsfw_r, reasoning_r) → float`: Weighted average

Bonus mechanisms:
- `confidence_bonus(confidence) → float`: +5% if confident and correct, -5% if overconfident and wrong

Configuration:
- `task_weights`: [0.25, 0.25, 0.25, 0.25]
- `confidence_bonus_scale`: 0.05

Example:
```python
shaper = RewardShaper()
episode_reward = shaper.compute_reward(0.9, 0.85, 0.80, 0.88)  # → 0.86
```

## Module 6: StateManager

The StateManager class tracks episode state and history.

Public methods:
- `create_observation(post, step, metadata) → Observation`: Formats post as observation
- `update_state(reward, decision, step)`: Logs action and reward
- `get_state() → dict`: Returns current episode state
- `reset_episode()`: Clear state for new episode

State tracking: current_step, episode_rewards, decisions_made, cumulative_reward

Configuration: max_steps=8, state_schema

Example:
```python
state_mgr = StateManager()
obs = state_mgr.create_observation(post, step=1, metadata={...})
state_mgr.update_state(reward=0.85, decision="reject", step=1)
state = state_mgr.get_state()  # → {step: 1, cumulative_reward: 0.85, ...}
```

## Integration Example

How modules work together in env.step():
```python
def step(self, action: Action) -> tuple:
  # 1. Get current post from ContentPool
  post = self.pool.get_batch()[self.batch_index]
  
  # 2. Get gold labels
  gold_labels = self.gold_labels[post["id"]]
  
  # 3. Grade agent decision
  tox_r = self.grader.grade_toxicity(action.confidence, gold_labels["toxicity"])
  spam_r = self.grader.grade_spam(action.decision == "reject", gold_labels["spam"])
  nsfw_r = self.grader.grade_nsfw(..., gold_labels["nsfw_category"])
  reason_r = self.grader.grade_reasoning(action.reasoning, gold_labels["justification"])
  
  # 4. Shape reward
  reward = self.reward_shaper.compute_reward(tox_r, spam_r, nsfw_r, reason_r)
  
  # 5. Update state
  self.state_mgr.update_state(reward, action.decision, self.current_step)
  
  # 6. Return observation, reward, done, info
  return obs, reward, done, info
```

## Testing Modules

Unit test structure:
```python
def test_grade_toxicity():
  grader = GraderModule()
  reward = grader.grade_toxicity(pred=0.8, gold=0.8)
  assert reward == 1.0
```

Integration test example: Mock gold labels and test full step.

## Extension Guide

To add a new task: Extend GraderModule with new grading method. To add a new classifier: Implement interface in ClassificationEngine. To modify reward weighting: Update task_weights in RewardShaper.

## Dependencies

Internal: ContentPool → StateManager, ClassificationEngine → RoutingEngine → GraderModule → RewardShaper. External: pydantic, asyncio, sentence-transformers, numpy.

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| toxicity_threshold_reject | 0.75 | Reject if toxicity > this |
| task_weights | [0.25, 0.25, 0.25, 0.25] | Weights for 4 tasks |
| similarity_model | "sentence-transformers/all-MiniLM-L6-v2" | Model for reasoning grading |