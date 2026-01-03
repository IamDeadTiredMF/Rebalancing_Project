# AI_USAGE.md
**Project:** Rebalancing_Project  
**Student:** Marius Fortuna  
**Course:** Advanced Programming (Fall 2025)

## Overview
I developed this project independently and used AI tools as a supporting resource (similar to documentation or online forums). AI assistance was limited to debugging, implementation support, and improving clarity of documentation. The project framing, financial reasoning, core methodology, and interpretation of results are my own.

## AI tools used
- ChatGPT (OpenAI)
- Claude (Anthropic)

## How AI tools were used
1. **Debugging and code review**
   - Help with diagnosing runtime errors and pandas index/alignment issues.
   - Sanity-checking that the full pipeline runs end-to-end via `python -m main`.

2. **Learning libraries and project structure**
   - Guidance on practical usage of `pathlib` for consistent paths and output folders.
   - Minor suggestions for keeping modules separated and organized.

3. **Implementation support for execution lag**
   - Assistance implementing the `pending` mechanism used to enforce a delay between signal generation and trade execution (to reduce look-ahead bias).

4. **ML trigger calibration idea**
   - Suggestion to use a rolling quantile trigger on predicted probabilities (instead of a fixed probability threshold) to reduce sensitivity to probability calibration drift.

5. **Documentation and writing**
   - Formatting improvements for README and minor grammar/clarity edits.

## What AI tools were not used for
- Defining the research question or overall approach.
- Selecting assets (QQQ/TLT), portfolio weights, or the train/test split concept.
- Designing the core label methodology and backtesting setup.
- Writing results interpretations or conclusions (these are based on the outputs produced by my code).