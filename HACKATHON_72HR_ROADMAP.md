# Content Moderation OpenEnv — 72-Hour Sprint Plan

**Problem:** Build an AI content moderation environment that classifies, flags, and routes user-generated content using an LLM agent.

**Real-world utility:** Every social platform (Reddit, Discord, X, YouTube) needs this.

**Your competitive advantage:** Integrate real API, multiple graded tasks, realistic reward shaping, reproducible baseline — all in 3 days.

---

## **Timeline: 3 Days × 24 Hours = 72 Hours**

### **DAY 1 (Hours 1–24): Foundation & Core Environment**

#### **Hours 1–3: Project Setup**
- [ ] Create GitHub repo: `content-moderation-openenv`
- [ ] Clone locally
- [ ] Set up directory structure:
  ```
  content-moderation-openenv/
  ├── openenv.yaml             # Core spec file
  ├── my_env.py                # Main environment class
  ├── inference.py             # Baseline script
  ├── Dockerfile
  ├── requirements.txt
  ├── README.md
  ├── data/
  │   ├── training_set.json    # Gold labels for grading
  │   └── synthetic_posts.json  # Real-world-like content
  └── graders/
      ├── toxicity_grader.py
      ├── spam_grader.py
      └── nsfw_grader.py
  ```
- [ ] Initialize venv: `python -m venv venv && source venv/bin/activate`
- [ ] Install basics: `pip install pydantic aiohttp openai python-dotenv`

#### **Hours 4–6: Generate Antigravity IDE Prompts for All .md Files**

Use these exact prompts in Antigravity IDE to generate each markdown file. **Copy-paste these into your IDE**.

**Prompt 1: openenv.yaml specification**
```
Generate a complete openenv.yaml file for a content moderation environment.

Requirements:
- name: content-moderation
- description: Brief real-world use case
- version: 1.0.0
- author: [Your Name]
- observation: Include content_text (string), content_id (uuid), timestamp
- action: Include decision (enum: approve/review/reject/escalate), reasoning (string)
- reward: float between 0.0 and 1.0
- tasks: Array with 4 tasks:
  1. toxicity_detection (detect harmful language)
  2. spam_classification (binary spam/ham)
  3. nsfw_detection (inappropriate content)
  4. reasoning_quality (justification quality)
- Each task has: name, description, difficulty (easy/medium/hard), evaluation_metric

Format as valid YAML with proper indentation.
```

**Prompt 2: README.md**
```
Write a comprehensive README for an OpenEnv content moderation environment.

Sections (in order):
1. Project Title & Badge
2. Problem Statement (2–3 sentences on real-world need)
3. Environment Overview (what the agent learns to do)
4. Action Space (what decisions the agent can make)
5. Observation Space (what the agent observes)
6. Task Descriptions
   - Task 1: Toxicity Detection (easy, goal is to flag toxic content with score)
   - Task 2: Spam Classification (medium, binary classification)
   - Task 3: NSFW Detection (hard, multi-class categorization)
   - Task 4: Reasoning Quality (bonus, check justification)
7. Reward Function (explain 0.0–1.0 scale, partial progress signals)
8. Setup & Installation (requirements.txt, docker commands)
9. Usage (how to reset(), step(), get baseline scores)
10. Baseline Performance (expected scores for each task)
11. Files & Structure
12. API Credentials Required (HF_TOKEN, API_BASE_URL, MODEL_NAME)

Tone: Professional, encouraging, actionable. Markdown formatting with code blocks.
```

**Prompt 3: AGENTS.md (for agent CLI tools)**
```
Generate AGENTS.md documentation for an RL agent to autonomously understand and use the content moderation environment.

Structure:
1. Environment Overview (in 1 paragraph)
2. State Space (exact JSON schema of observations)
3. Action Space (exact enum values: approve, review, reject, escalate + reasoning field)
4. Step Return Format (observation, reward, done, info)
5. Success Criteria for Each Task
   - Toxicity: accuracy vs gold labels + score calibration
   - Spam: F1-score on binary classification
   - NSFW: macro F1-score on 4-class problem
   - Reasoning: BLEU/ROUGE score vs reference justifications
6. Initialization (reset() return format)
7. Episode Termination (max steps = 8)
8. Reward Shaping Strategy (how partial progress is scored)
9. Common Agent Patterns (classify then route, iterative refinement)
10. Example Trajectories (2–3 annotated step sequences)

Format: Markdown, code blocks for schemas, YAML for config examples.
```

**Prompt 4: TASKS.md**
```
Generate detailed task specifications for a content moderation OpenEnv.

For each of 4 tasks, include:
- Task Name & ID
- Difficulty (easy/medium/hard) with reasoning
- Objective (1–2 sentence)
- Success Metric (accuracy, F1, BLEU, etc.)
- Examples (3–5 real-world content samples)
- Grading Logic (pseudo-code for how reward is computed)
- Baseline Agent Strategy (what a naive agent would do)

Tasks:
1. Toxicity Detection
   - Easy: Detect presence of toxic/hateful language
   - Metric: Accuracy vs gold labels + sigmoid-smoothed confidence
   
2. Spam Classification
   - Medium: Distinguish promotional/malicious spam from legitimate posts
   - Metric: F1-score (spam is rare, so recall matters)
   
3. NSFW Content Detection
   - Hard: Categorize inappropriate content (violence, adult, explicit text, etc.)
   - Metric: Macro F1 across 4 categories
   
4. Reasoning Quality
   - Bonus: Quality of the agent's justification (not just decision)
   - Metric: Embedding similarity to reference justifications (sentence-BERT)

Format: Markdown with tables for metrics, code blocks for examples.
```

**Prompt 5: MODULES.md**
```
Generate MODULES.md documenting the internal structure for autonomous agents.

Sections:
1. Core Modules Overview
2. ContentPool Module
   - Loads synthetic + real posts
   - Rotates through batches
   - API integration notes (Reddit/HF API)
3. ClassificationEngine
   - Toxicity classifier internals
   - Spam detector logic
   - NSFW detector logic
4. RoutingEngine
   - Decision tree for approve/review/reject/escalate
   - Confidence thresholds
5. GraderModule
   - Gold label matching
   - Accuracy computation
   - F1-score logic
6. RewardShaper
   - Combines 4 task rewards into episode reward
   - Bonus for reasoning quality
7. StateManager
   - Observation generation
   - Episode tracking
   - History management

Each module includes:
- Class signature
- Key methods
- Input/output types
- Dependencies
- Tuning parameters (thresholds, weights)

Format: Code-like pseudocode + plain English explanations.
```

#### **Hours 7–12: Implement Core Environment (my_env.py)**

**Antigravity prompt for step-by-step code generation:**
```
Generate a complete Python async OpenEnv-compliant content moderation environment.

Class: ContentModerationEnv

Requirements:
1. Pydantic models:
   - Observation: content_id, content_text, source, timestamp, metadata (dict)
   - Action: decision (approve/review/reject/escalate), reasoning (str), confidence (0–1)
   - Reward: float 0.0–1.0
   
2. Async methods:
   - async def reset() → Observation, done=False
   - async def step(action: Action) → (Observation, Reward, done, info)
   - async def state() → dict with episode stats
   - async def close()
   
3. Episode flow:
   - Load batch of ~8 posts from synthetic dataset
   - Agent classifies each post (1 step per post, max 8 steps)
   - Reward based on accuracy + reasoning quality
   - Episode ends when all posts processed or max steps reached
   
4. Integration:
   - Load gold labels from data/training_set.json
   - Compute grading for each decision
   - Accumulate rewards across episode
   
5. Async patterns:
   - Use asyncio.sleep(0) for yields if needed
   - Support from_docker_image() and from_env() initialization
   
6. State tracking:
   - current_step, episode_rewards, decisions_made, accuracy_so_far

Include type hints, docstrings, error handling. Make it production-ready.
```

#### **Hours 13–18: Implement Graders (graders/*.py)**

Create 3 grader modules: toxicity, spam, nsfw.

**Antigravity prompt for graders:**
```
Generate 3 grader modules for content moderation OpenEnv.

Module 1: toxicity_grader.py
- Function: grade_toxicity(content: str, predicted_score: float, gold_label: str) → float reward
- Logic:
  - If agent predicted toxic and gold label is toxic → reward approaches 1.0 (accuracy + confidence bonus)
  - If agent missed toxic content → reward 0.0
  - If false positive → reward 0.3
  - Confidence calibration: if pred_score ∈ (0.6, 0.8) but label suggests 0.9, small penalty
- Return: 0.0–1.0 reward

Module 2: spam_grader.py
- Function: grade_spam(content: str, predicted_class: bool, gold_label: bool) → float reward
- Logic:
  - F1-score basis: TP/(TP + 0.5*(FP+FN))
  - Recall weight 2x because missing spam is worse than false flagging
- Return: 0.0–1.0 reward

Module 3: nsfw_grader.py
- Function: grade_nsfw(content: str, predicted_category: str, gold_category: str) → float reward
- Logic:
  - Perfect match (predicted == gold) → 1.0
  - One-off category (confusing "violence" with "explicit") → 0.6
  - Completely wrong → 0.0
- Return: 0.0–1.0 reward

All use deterministic, reproducible logic. Include docstrings + test cases.
```

#### **Hours 19–24: Data Prep & Synthetic Posts**

**Create data/training_set.json:**
```
Antigravity prompt:
Generate 50 realistic social media posts with moderation labels.

Format (JSON array of objects):
[
  {
    "id": "post_001",
    "content": "Great weather today!",
    "source": "twitter",
    "toxicity": 0.0,
    "spam": false,
    "nsfw_category": "safe",
    "justification": "Generic positive post, no moderation flags"
  },
  ...
]

Include:
- 10 clean posts (no flags)
- 10 toxic posts (varying severity 0.4–0.9)
- 10 spam posts (promo, scams, bot-like)
- 10 NSFW posts (violence, explicit, adult themes)
- 10 borderline/ambiguous (challenge agent reasoning)

For each post:
- toxicity: float 0.0–1.0 (severity of harmful language)
- spam: boolean
- nsfw_category: "safe" | "violence" | "explicit" | "adult_content"
- justification: why this label (agent learns from this)

Make posts realistic, diverse, non-offensive to humans (clearly labeled as test data).
```

---

### **DAY 2 (Hours 25–48): Integration & Baseline Script**

#### **Hours 25–30: Implement inference.py**

Copy the template from the requirements document and adapt:

**Antigravity prompt:**
```
Generate a complete inference.py for content moderation OpenEnv using OpenAI client.

Requirements:
- Load env variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
- Initialize OpenAI client
- Async main() function
- Create env instance (ContentModerationEnv)
- Reset env, run 8-step episode
- For each step:
  1. Build user prompt: "Classify this post: {content}. Output: decision, reasoning, confidence"
  2. Call LLM (Qwen/other model)
  3. Parse response as Action (decision, reasoning, confidence)
  4. Call env.step(action)
  5. Collect reward
  6. Log [STEP] line with exact format from requirements
- After episode: log [END] with success=true/false, steps, rewards
- Emit [START] at beginning with task name + model name
- Reproduce consistent scores (run multiple times, average)

Logging format MUST match:
[START] task=content_moderation env=openenv_moderation model={MODEL_NAME}
[STEP] step={n} action="{decision}:{reasoning[:20]}" reward=X.XX done=false error=null
[END] success=true steps=8 rewards=R1,R2,...,R8

Include error handling, retries, timeout protection.
```

#### **Hours 31–36: Docker & Deployment Prep**

**Create Dockerfile:**
```
Antigravity prompt:
Generate a Dockerfile for content moderation OpenEnv.

Requirements:
- Base: python:3.11-slim
- Copy requirements.txt, install dependencies
- Copy entire project into /app
- Set working directory to /app
- Expose port 8000 (for HF Space health check)
- Default command: python inference.py
- Include environment variable defaults for API_BASE_URL, MODEL_NAME
- Ensure all dependencies (aiohttp, pydantic, openai) are installed

Minimize layer count, use .dockerignore for __pycache__ / .git / venv
```

**Create requirements.txt:**
```
pydantic==2.5.0
aiohttp==3.9.1
openai==1.3.0
python-dotenv==1.0.0
```

#### **Hours 37–42: openenv.yaml & Validation**

**Antigravity prompt for openenv.yaml:**
```
Generate a complete, validation-ready openenv.yaml for content moderation.

Structure:
name: content-moderation
version: 1.0.0
description: "Autonomous content moderation agent learns to classify, flag, and route user-generated content across toxicity, spam, and NSFW categories."
author: "Your Name"

observation:
  type: object
  required: [content_id, content_text, source, timestamp]
  properties:
    content_id:
      type: string
      description: "Unique post ID"
    content_text:
      type: string
      description: "Full text of the post to moderate"
    source:
      type: string
      enum: ["twitter", "reddit", "discord"]
    timestamp:
      type: string
      format: "iso8601"
    metadata:
      type: object
      additionalProperties: true

action:
  type: object
  required: [decision, reasoning]
  properties:
    decision:
      type: string
      enum: ["approve", "review", "reject", "escalate"]
    reasoning:
      type: string
      maxLength: 200
    confidence:
      type: number
      minimum: 0.0
      maximum: 1.0

reward:
  type: number
  minimum: 0.0
  maximum: 1.0
  description: "Task accuracy score"

tasks:
  - id: "toxicity_detection"
    name: "Toxicity Detection"
    description: "Classify posts for harmful/hateful language, score severity 0–1"
    difficulty: "easy"
    success_metric: "accuracy + confidence calibration"
    
  - id: "spam_classification"
    name: "Spam Classification"
    description: "Binary classification: spam or legitimate"
    difficulty: "medium"
    success_metric: "F1-score (spam rare)"
    
  - id: "nsfw_detection"
    name: "NSFW Content Detection"
    description: "Categorize: violence, explicit, adult_content, safe"
    difficulty: "hard"
    success_metric: "macro F1-score"
    
  - id: "reasoning_quality"
    name: "Reasoning Quality"
    description: "Quality of agent justification"
    difficulty: "medium"
    success_metric: "semantic similarity to reference"

Include proper YAML formatting, no tabs, 2-space indents.
```

#### **Hours 43–48: Testing & Baseline Run**

- [ ] Run `python inference.py` locally (with mock env first)
- [ ] Verify [START], [STEP], [END] logs are properly formatted
- [ ] Collect 3 baseline runs, average scores
- [ ] Document expected baseline scores in README (e.g., "Naive agent: ~0.55 avg reward")

---

### **DAY 3 (Hours 49–72): Polish, Deployment, Final Checks**

#### **Hours 49–54: HF Space Setup**

- [ ] Create HF Space: `https://huggingface.co/spaces/{your-username}/content-moderation-openenv`
- [ ] Choose "Docker" runtime
- [ ] Push repo to HF (with Dockerfile at root)
- [ ] HF auto-builds and deploys
- [ ] Verify Space URL responds with 200 status
- [ ] Run `openenv validate` via HF Space endpoint (if supported) or manually

#### **Hours 55–60: Final Documentation & Polish**

- [ ] Complete README.md with all sections
- [ ] Add setup instructions: `pip install -r requirements.txt`, `python inference.py`
- [ ] Document expected baseline scores (table format)
- [ ] Add "Known Limitations" section
- [ ] Add "Future Work" (multi-modal, real API, streaming graders)
- [ ] Create .dockerignore file
- [ ] Add MIT license

**Antigravity prompt for polished README:**
```
Refine README.md to competition-ready standard.

Add:
1. "Quick Start" section (3-line setup)
2. "Baseline Performance" table:
   | Model | Toxicity | Spam | NSFW | Reasoning | Avg |
   | Qwen2.5-72B | 0.82 | 0.76 | 0.68 | 0.71 | 0.74 |
3. "Evaluation Criteria Met" checklist
4. "Project Highlights" (what makes this special)
5. "Citation" (if submitting to a conference)

Keep tone encouraging. Make it clear this solves real problems.
```

#### **Hours 61–66: Pre-Submission Checklist**

- [ ] **HF Space deploys** – ping URL, gets 200 response
- [ ] **Dockerfile builds** – `docker build -t content-mod .` works locally
- [ ] **inference.py runs** – produces [START], [STEP], [END] logs exactly
- [ ] **openenv.yaml valid** – all required fields present, proper YAML
- [ ] **4 tasks with graders** – toxicity, spam, nsfw, reasoning all callable + return 0.0–1.0
- [ ] **Baseline reproduces** – run inference.py 3x, scores within ±0.05
- [ ] **README complete** – environment description, tasks, setup, baseline
- [ ] **All .md files generated** – openenv.yaml, README.md, AGENTS.md, TASKS.md, MODULES.md

#### **Hours 67–72: Final Submission & Buffer**

- [ ] Push final commit to GitHub
- [ ] Verify HF Space auto-updated
- [ ] Run full end-to-end test (reset → 8 steps → grade → log)
- [ ] Document any known issues in README
- [ ] **SUBMIT** before deadline
- [ ] Buffer time: Unexpected failures, last-minute tweaks, docker issues

---

## **Key Decision Points (Ask Yourself)**

1. **Real API or Synthetic?**
   - ✅ Use Reddit/HuggingFace post API for realistic data (do this)
   - This impresses judges and reflects real-world value

2. **Model Choice for Baseline?**
   - Use Qwen2.5-72B (on HF router) or Claude (if you have credits)
   - Qwen is free via HF, try it first

3. **Reward Shaping Strategy?**
   - Simple: accuracy on gold labels (0.0–1.0)
   - Better: accuracy + confidence calibration + partial credit for close calls
   - Use **better** approach (it shows thoughtful design)

4. **Episode Length?**
   - 8 posts per episode (as in template) is good
   - Could do 4 (faster) or 16 (harder agent challenge)
   - Stick with 8 for reproducibility

5. **Grading Complexity?**
   - Deterministic rule-based (toxicity: gold vs pred match)
   - vs. Semantic-similarity-based (embedding cosine for reasoning)
   - Use **deterministic first**, add semantic if you have time

---

## **Antigravity IDE Workflow (Fast Setup)**

For each `.md` file you need to generate:

1. Open Antigravity IDE
2. Paste the exact prompt from "Hours 4–6" section above
3. Let it generate (takes 2–5 min per file)
4. Copy output → paste into your repo as `.md` file
5. Light edits (personalize author name, fix any typos)
6. Commit to GitHub

**This cuts documentation time from 6 hours → 1 hour.**

---

## **Competitive Edge Checklist**

✅ **Real-world problem** (content moderation, not toy)  
✅ **Multiple tasks** (4 tasks with clear difficulty ladder)  
✅ **Smart reward shaping** (accuracy + confidence + reasoning quality)  
✅ **Real API integration** (Reddit/HF posts, not just synthetic)  
✅ **Clean code** (typed, async, production-grade)  
✅ **Reproducible baseline** (consistent scores across runs)  
✅ **Full deployment** (Docker + HF Space + working endpoints)  
✅ **Comprehensive docs** (README, AGENTS.md, TASKS.md, MODULES.md)  

---

## **If You Get Stuck**

| Issue | Solution | Time |
|-------|----------|------|
| Environment won't reset | Check Pydantic model fields match observation schema | 15 min |
| inference.py parsing fails | Wrap LLM response in regex to extract decision/reasoning | 20 min |
| Grader always returns 0.0 | Verify gold labels are loading correctly, mock grader returns 0.5 | 10 min |
| Dockerfile build fails | Missing `openai` in requirements.txt | 5 min |
| HF Space won't deploy | Check `.gitignore` not hiding Dockerfile, push again | 10 min |
| Baseline scores too low | Model is undertrained; use a larger model or refine prompts | 30 min |

---

## **Final Word**

You have **exactly 72 hours**. This roadmap is minute-optimized.

- **Don't add extra features.** Focus on the core 4 tasks + clean graders.
- **Don't refactor.** If it works, ship it.
- **Do test end-to-end** (reset → step → grade → log) at hour 60.
- **Do keep a git log** so you can revert if something breaks at hour 70.

**You've got this.** Go build something real. 🚀
