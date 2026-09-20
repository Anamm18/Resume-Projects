"""
Autonomous Data Analysis Agent
================================
A LangGraph-based agent that autonomously plans, writes, and executes
pandas code to analyze a CSV file, then produces a plain-English
insights report — without a human specifying each analysis step.

Usage:
    export OPENROUTER_API_KEY="your-key-here"
    python agent.py --csv path/to/data.csv --output report.md
"""

import os
import re
import time
import argparse
import io
import contextlib
import pandas as pd
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Loads OPENROUTER_API_KEY / OPENROUTER_MODEL from a .env file into the
# environment (for CLI usage; app.py loads it too for the Streamlit UI).
# We point explicitly at the .env next to THIS file rather than relying on
# the current working directory, since that depends on how/where you
# launched the script from and is a common source of "key not found" bugs.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_ENV_PATH)


# ---------------------------------------------------------------------------
# State: carried between every node in the graph
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    df_path: str
    plan: List[str]
    completed: List[str]
    insights: List[str]
    iterations: int
    max_iterations: int


# OpenRouter is OpenAI-API-compatible, so ChatOpenAI works by pointing it
# at OpenRouter's base_url with your OpenRouter key.
# Swap the model to any slug from https://openrouter.ai/models, e.g.
# "anthropic/claude-3.5-sonnet", "openai/gpt-4o-mini",
# "meta-llama/llama-3.1-8b-instruct:free" (free tier, weaker quality)
llm = None  # initialized lazily via init_llm() so callers (CLI or Streamlit)
            # can supply the API key at runtime instead of only via env var


def init_llm(api_key: str = None, model: str = None, max_tokens: int = None):
    """Must be called once before running the agent (build_agent/run_agent
    call this automatically if you pass api_key/model through, or you can
    call it yourself first)."""
    global llm
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "No OpenRouter API key provided. Pass api_key= or set "
            "the OPENROUTER_API_KEY environment variable."
        )
    model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")

    # IMPORTANT: without an explicit max_tokens, ChatOpenAI lets the model
    # default to its maximum possible output (e.g. 65,536 tokens for some
    # Claude models). OpenRouter reserves credit up front based on that
    # ceiling, not on what actually gets generated — so on a low/free
    # balance you'll get a 402 "requires more credits" error even though
    # the actual response would've been tiny. Our prompts only ever need a
    # short plan, a few lines of code, or a short report, so a modest cap
    # is plenty and avoids that entirely.
    max_tokens = max_tokens or int(os.getenv("OPENROUTER_MAX_TOKENS", "1024"))

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    return llm


def invoke_with_retry(prompt: str, max_retries: int = 3, base_wait: int = 15):
    """
    Wraps llm.invoke() with retries for OpenRouter's transient 402 errors
    (e.g. "in_flight_budget_exhausted" on low/free-tier accounts, which
    clears itself once earlier requests finish — see Retry-After).
    Raises the underlying error if it isn't one of these, or once retries
    are exhausted.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            msg = str(e)
            is_retryable = "in_flight_budget_exhausted" in msg or "429" in msg
            last_error = e
            if not is_retryable or attempt == max_retries:
                raise
            # Try to honor a Retry-After value embedded in the error message;
            # otherwise back off with a growing wait.
            match = re.search(r"Retry-After['\"]?:\s*['\"]?(\d+)", msg)
            wait = int(match.group(1)) if match else base_wait * attempt
            print(f"[RETRY] Attempt {attempt}/{max_retries} hit a transient "
                  f"error, waiting {wait}s before retrying: {msg[:150]}")
            time.sleep(wait)
    raise last_error


# ---------------------------------------------------------------------------
# Tool: executes agent-generated pandas code safely-ish and captures output
# ---------------------------------------------------------------------------
def run_pandas_code(code: str, df: pd.DataFrame) -> str:
    local_vars = {"df": df, "pd": pd}
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"pd": pd}, local_vars)
        output = buffer.getvalue().strip()
        return output if output else "Code ran with no printed output."
    except Exception as e:
        return f"Error executing code: {e}"


def clean_code_block(raw: str) -> str:
    """Strips markdown code fences if the LLM adds them despite instructions."""
    code = raw.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        code = "\n".join(lines)
    return code.strip()


# ---------------------------------------------------------------------------
# Node 1: Plan — agent inspects the data and decides what to analyze
# ---------------------------------------------------------------------------
def plan_node(state: AgentState) -> AgentState:
    df = pd.read_csv(state["df_path"])
    summary = (
        f"Columns: {list(df.columns)}\n"
        f"Shape: {df.shape}\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n"
        f"Sample rows:\n{df.head(3).to_string()}"
    )

    prompt = f"""You are a senior data analyst planning an exploratory data
analysis. Given this dataset summary:

{summary}

List 4-6 concrete, specific analysis steps you would perform (e.g. "check
missing values per column", "distribution of the 'price' column",
"correlation matrix of numeric columns", "top 5 categories in 'region' by
count"). Return ONLY a numbered list, one step per line. No preamble."""

    response = invoke_with_retry(prompt)
    steps = []
    for line in response.content.strip().split("\n"):
        line = line.strip()
        if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
            steps.append(line.split(".", 1)[-1].strip())
        elif line:
            steps.append(line)

    state["plan"] = steps
    state["completed"] = []
    state["iterations"] = 0
    print(f"[PLAN] Agent generated {len(steps)} analysis steps.")
    return state


# ---------------------------------------------------------------------------
# Node 2: Execute — agent writes and runs code for the next planned step
# ---------------------------------------------------------------------------
def execute_step_node(state: AgentState) -> AgentState:
    df = pd.read_csv(state["df_path"])
    step_index = len(state["completed"])
    next_step = state["plan"][step_index]

    code_prompt = f"""Write ONLY executable Python pandas code (no
explanation, no markdown fences) to accomplish this analysis step on a
dataframe called `df`:

"{next_step}"

Use print() so results are captured as text output. Keep it to a few lines."""

    response = invoke_with_retry(code_prompt)
    code = clean_code_block(response.content)

    result = run_pandas_code(code, df)
    state["completed"].append(
        f"### Step {step_index + 1}: {next_step}\n"
        f"```python\n{code}\n```\n"
        f"**Result:**\n```\n{result}\n```"
    )
    state["iterations"] += 1
    print(f"[EXECUTE] Step {step_index + 1}/{len(state['plan'])} done.")
    return state


# ---------------------------------------------------------------------------
# Conditional edge: the agentic decision — keep going or wrap up?
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> str:
    if (len(state["completed"]) >= len(state["plan"])
            or state["iterations"] >= state["max_iterations"]):
        return "summarize"
    return "continue"


# ---------------------------------------------------------------------------
# Node 3: Summarize — turn raw results into a stakeholder-readable report
# ---------------------------------------------------------------------------
def summarize_node(state: AgentState) -> AgentState:
    all_results = "\n\n".join(state["completed"])
    prompt = f"""Based on this raw exploratory data analysis output, write a
concise insights report as 5-8 bullet points that a non-technical
stakeholder could understand. Focus on what matters — trends, anomalies,
data quality issues, and actionable observations. No jargon.

Raw analysis:
{all_results}"""

    response = invoke_with_retry(prompt)
    state["insights"] = [
        line.strip() for line in response.content.strip().split("\n") if line.strip()
    ]
    print("[SUMMARIZE] Insights report generated.")
    return state


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_step_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges(
        "execute",
        should_continue,
        {"continue": "execute", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_agent(csv_path: str, output_path: str, max_iterations: int = 6,
              api_key: str = None, model: str = None):
    if llm is None:
        init_llm(api_key, model)
    agent = build_agent()

    initial_state = {
        "df_path": csv_path,
        "plan": [],
        "completed": [],
        "insights": [],
        "iterations": 0,
        "max_iterations": max_iterations,
    }

    final_state = agent.invoke(initial_state)

    report_lines = ["# Data Analysis Insights Report\n"]
    report_lines.append("## Analysis Plan\n")
    for step in final_state["plan"]:
        report_lines.append(f"- {step}")

    report_lines.append("\n## Detailed Steps & Results\n")
    report_lines.extend(final_state["completed"])

    report_lines.append("\n## Key Insights\n")
    report_lines.extend(final_state["insights"])

    report_text = "\n".join(report_lines)
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to {output_path}")
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Data Analysis Agent")
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, default="report.md",
                         help="Path to save the markdown report")
    parser.add_argument("--max-iterations", type=int, default=6)
    args = parser.parse_args()

    init_llm()  # reads OPENROUTER_API_KEY / OPENROUTER_MODEL from env
    run_agent(args.csv, args.output, args.max_iterations)
