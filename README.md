# Multi-Modal Evidence Review System

An automated insurance claim verification system that analyzes multi-modal inputs (claim conversation transcript, user historical context, and image evidence) to verify claim validity.

## Architecture

- **`output.csv`**: Contains the structured results matching the required evaluation schema.
- **`code/`**:
  - `claim_processor.py`: Core system parsing claim targets, loading/scaling images, executing structured prompts against the VLM, and sanitizing output categories.
  - `main.py`: Entry point loading datasets, triggering VLM reviews sequentially with rate-limit buffers, logging activities, and exporting results.
  - `config.py`: Threshold parameters, valid classifications (car/laptop/package parts), and default system constants.

## Features

- Fully parses claim objects, parts, and damage categories directly from conversation history.
- Resolves conflicting details and evaluates image evidence standards.
- Flags suspicious user historical indicators dynamically (`user_history_risk`, `manual_review_required`).
- Handles VLM rate limits dynamically with retry intervals.
