# Task Specifications

## 1. Toxicity Detection (toxicity_detection)
- **Goal**: Identify harmful language across a spectrum from minor insults to severe hate speech.
- **Input**: `content_text`
- **Output**: `toxicity_score` (0.0 - 1.0)
- **Evaluation**: Distance from gold label score.

## 2. Spam Classification (spam_classification)
- **Goal**: Detect automated promotional or malicious spam.
- **Input**: `content_text`
- **Output**: `is_spam` (Boolean)
- **Evaluation**: F1-Score focusing on recall (catching all spam).

## 3. NSFW Detection (nsfw_detection)
- **Goal**: Categorize content into specific inappropriate buckets.
- **Categories**: `safe`, `violence`, `explicit`, `adult_content`.
- **Input**: `content_text`
- **Evaluation**: Macro F1-Score across all categories.

## 4. Reasoning Quality (reasoning_quality)
- **Goal**: Ensure that decisions are backed by logical guidelines.
- **Input**: `reasoning` string.
- **Evaluation**: Guideline adherence and semantic clarity.