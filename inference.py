# inference.py
import asyncio
import os
import json
from typing import List, Optional
from openai import OpenAI
from my_env import ContentModerationEnv, Action
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

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

Be decisive but calibrated. Follow the JSON format strictly."""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action_short: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
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
        
        # Clean potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
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
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY environment variable not set.")
        return

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
            
            action_short = f"{action.decision}:{action.reasoning[:20]}"
            log_step(step=step, action_short=action_short, reward=reward, done=done, error=None)
            
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
