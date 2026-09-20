# Autonomous Data Analysis Agent

An agentic AI system built with **LangGraph** that autonomously plans, writes, and executes its own pandas code to explore any CSV dataset — then generates a plain-English insights report, with no human specifying the analysis steps.

This is not a chatbot wrapper: the agent follows a **plan → act → observe → decide** loop, where it decides for itself when the analysis is complete.

## How it works

1. **Plan node** — the agent inspects the dataset's columns, shape, and dtypes, and generates a list of analysis steps to perform (missing values, distributions, correlations, group-bys, etc.)
2. **Execute node** — for each planned step, the agent writes pandas code on the fly and executes it, capturing the output
3. **Conditional edge** — after each step, the agent checks whether all planned steps are done (or a max-iteration safety limit is hit) and decides whether to continue or move to summarizing
4. **Summarize node** — the agent turns all the raw execution output into a concise, non-technical insights report

```
        ┌──────┐
        │ plan │
        └──┬───┘
           ▼
      ┌─────────┐
   ┌─▶│ execute │
   │  └────┬────┘
   │       ▼
   │  [continue?] ──yes──┘
   │       │no
   │       ▼
   │  ┌───────────┐
   └──┤ summarize │
      └───────────┘
```

## Tech Stack
Python, LangGraph, LangChain, OpenRouter (LLM gateway), pandas

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```
Then edit `.env` and paste in your real OpenRouter API key. Both `app.py`
and `agent.py` load `.env` automatically (via `python-dotenv`) — the
Streamlit sidebar will show "Using OpenRouter API key from .env" instead of
asking you to type it in each time.

By default this uses `anthropic/claude-sonnet-5` via OpenRouter. To use a
different model (e.g. a free-tier one for testing), set `OPENROUTER_MODEL`
in your `.env` file, e.g.:
```
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free
```
See https://openrouter.ai/models for all available model slugs and pricing
— model availability changes over time, so if you hit a 404 "no endpoints
found" error, check that page for the current slug.

### Troubleshooting OpenRouter errors

- **402 "requires more credits" (max_tokens related):** fixed by the
  `OPENROUTER_MAX_TOKENS` setting above — OpenRouter reserves credit based
  on the *maximum possible* output, not what's actually used.
- **402 "in_flight_budget_exhausted":** a separate limit on free/very-low
  balance accounts — it caps how much total value can be mid-request at
  once, independent of `max_tokens`. The agent automatically retries this
  a few times with backoff (see `invoke_with_retry` in `agent.py`), so a
  brief pause (up to ~2 minutes) during a run is expected and not a bug.
  If it keeps failing, either add a small amount of credit (this raises
  the ceiling) or switch to a `:free` model slug, which isn't subject to
  the paid in-flight budget at all.

## Usage

### Option A — Streamlit app (recommended for demos/screenshots)
```bash
streamlit run app.py
```
Upload a CSV, enter your OpenRouter key in the sidebar, and watch the agent's plan and each step stream in live, with a downloadable markdown report at the end. This is the best format for a resume demo GIF/screenshot.

### Option B — Command line
```bash
python agent.py --csv sample_data.csv --output report.md
```
Or on your own CSV:
```bash
python agent.py --csv path/to/your_data.csv --output report.md --max-iterations 8
```
Prints progress to the terminal and saves a full markdown report (plan, code + results per step, final insights) to the output file.



## Example Output

Running on `sample_data.csv`, the agent might autonomously decide to:
- Check missing values (finds gaps in `discount` and `ship_mode`)
- Flag the outlier order and how much it skews total revenue
- Break down profit by category and region, and flag any loss-making segments
- Check for duplicate rows

...all without being told which of these to check — it decides based on what the data actually looks like.

## Notes / Future Improvements
- Add a "critique" node that reviews each step's output before moving on, and can re-plan if a step reveals something unexpected
- Sandbox code execution properly for production use (e.g. `RestrictedPython` or a subprocess with restricted builtins) instead of raw `exec()`
- Add a charting tool (matplotlib) so the agent can generate visualizations, not just text output
- Swap the step-counter stopping condition for an LLM self-assessment of "do I have enough to report?"
