"""
Streamlit UI for the Autonomous Data Analysis Agent.

Usage:
    streamlit run app.py
"""

import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import build_agent, init_llm

# Loads OPENROUTER_API_KEY / OPENROUTER_MODEL from the .env file next to
# this script. Pointing at an explicit path (rather than a bare
# load_dotenv(), which depends on the current working directory) avoids
# "key not found" issues caused by launching Streamlit from elsewhere.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_ENV_PATH)

st.set_page_config(page_title="Autonomous Data Analyst", layout="wide")

st.title("🤖 Autonomous Data Analysis Agent")
st.write(
    "Upload a CSV. The agent inspects it, **plans its own analysis steps**, "
    "writes and runs pandas code for each one, and produces a plain-English "
    "insights report — no steps specified by you."
)

# ---------------------------------------------------------------------------
# Sidebar: API settings
# ---------------------------------------------------------------------------
env_api_key = os.getenv("OPENROUTER_API_KEY", "")
env_model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")

with st.sidebar:
    st.header("Settings")

    if env_api_key:
        st.success("✅ Using OpenRouter API key from .env")
        api_key = env_api_key
        if st.checkbox("Use a different key for this session"):
            api_key = st.text_input("OpenRouter API Key", type="password")
    else:
        st.warning("No OPENROUTER_API_KEY found in .env")
        with st.expander("Why am I seeing this?"):
            st.write(f"Looked for a `.env` file at:\n\n`{_ENV_PATH}`")
            st.write(
                "Checklist:\n"
                "- Does a file with exactly that path exist?\n"
                "- On Windows, check it isn't secretly named `.env.txt` "
                "(File Explorer hides known extensions by default — "
                "enable 'File name extensions' in the View tab to check)\n"
                "- Does the file contain a line like "
                "`OPENROUTER_API_KEY=sk-or-...` with no quotes and no "
                "extra spaces around the `=`?\n"
                "- Do you also have `OPENROUTER_API_KEY` set as a real "
                "Windows environment variable (e.g. via `setx` or System "
                "Properties)? If so, an empty/wrong value there can "
                "override the .env file's value."
            )
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get one at https://openrouter.ai/keys. To avoid typing "
                 "this every time, create a .env file next to app.py with "
                 "a line: OPENROUTER_API_KEY=your-key-here",
        )

    model = st.text_input(
        "Model",
        value=env_model,
        help="Any model slug from https://openrouter.ai/models, e.g. "
             "anthropic/claude-sonnet-5, openai/gpt-4o-mini, or "
             "meta-llama/llama-3.1-8b-instruct:free (free tier)",
    )
    max_iterations = st.slider("Max analysis steps", 3, 10, 6)
    max_tokens = st.slider(
        "Max tokens per LLM call", 256, 4096,
        int(os.getenv("OPENROUTER_MAX_TOKENS", "1024")), step=256,
        help="Keep this low on a free/low-credit OpenRouter account — a "
             "high value reserves that much credit upfront even if the "
             "model uses far less, which causes 402 'insufficient "
             "credits' errors.",
    )
    st.caption(
        "Tip: use a `:free` model slug to test without spending credits, "
        "then switch to a stronger model for your final resume demo."
    )

# ---------------------------------------------------------------------------
# Main: upload + preview
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df_preview.head(10), use_container_width=True)
    st.caption(f"{df_preview.shape[0]} rows × {df_preview.shape[1]} columns")

    run_clicked = st.button("🚀 Run Autonomous Analysis", type="primary",
                             disabled=not api_key)
    if not api_key:
        st.warning("Enter your OpenRouter API key in the sidebar to run the agent.")

    if run_clicked:
        # Save the upload to a temp file — the agent reads from a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            uploaded_file.seek(0)
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            init_llm(api_key=api_key, model=model, max_tokens=max_tokens)
            agent = build_agent()

            initial_state = {
                "df_path": tmp_path,
                "plan": [],
                "completed": [],
                "insights": [],
                "iterations": 0,
                "max_iterations": max_iterations,
            }

            plan_placeholder = st.empty()
            steps_header = st.empty()
            progress_bar = st.progress(0.0)
            final_state = None

            # .stream() yields the updated state after each node runs, so we
            # can show the agent's plan, then each step, as they happen —
            # rather than waiting for the whole run to finish.
            with st.spinner(
                "Agent is working... (if your OpenRouter balance is low, a "
                "step may pause up to ~2 minutes to retry automatically "
                "after a temporary rate limit — this is normal, not a hang)"
            ):
                for update in agent.stream(initial_state):
                    for node_name, node_state in update.items():
                        if node_name == "plan":
                            with plan_placeholder.container():
                                st.subheader("📋 Analysis Plan (agent-generated)")
                                for i, step in enumerate(node_state["plan"], 1):
                                    st.markdown(f"{i}. {step}")
                            steps_header.subheader("🔍 Steps")

                        elif node_name == "execute":
                            completed = node_state["completed"]
                            with st.expander(f"Step {len(completed)}", expanded=False):
                                st.markdown(completed[-1])
                            progress_bar.progress(
                                min(len(completed) / max_iterations, 1.0)
                            )

                        elif node_name == "summarize":
                            final_state = node_state

            progress_bar.progress(1.0)

            if final_state:
                st.subheader("💡 Key Insights")
                for line in final_state["insights"]:
                    st.markdown(f"- {line}" if not line.startswith(("-", "*")) else line)

                report_lines = ["# Data Analysis Insights Report\n"]
                report_lines.append("## Analysis Plan\n")
                report_lines += [f"- {s}" for s in final_state["plan"]]
                report_lines.append("\n## Detailed Steps & Results\n")
                report_lines += final_state["completed"]
                report_lines.append("\n## Key Insights\n")
                report_lines += final_state["insights"]
                report_text = "\n".join(report_lines)

                st.download_button(
                    "📥 Download Full Report (Markdown)",
                    data=report_text,
                    file_name="insights_report.md",
                    mime="text/markdown",
                )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
        finally:
            os.remove(tmp_path)

else:
    st.info(
        "No CSV yet? Try the included `sample_data.csv` (a synthetic sales "
        "dataset with realistic messiness — missing values, an outlier, a "
        "duplicate row) or download a larger real-world dataset — see the "
        "README for links."
    )
