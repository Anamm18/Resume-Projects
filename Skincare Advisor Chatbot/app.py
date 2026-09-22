"""
Skincare Ingredient Recommendation Chatbot — Streamlit app.

Wraps the retrieval-augmented pipeline from the notebook:
  1. Embed the user's query with sentence-transformers (all-MiniLM-L6-v2)
  2. Retrieve nearest ingredients from a persistent ChromaDB collection
  3. Optionally filter candidates by mentioned skin type
  4. Generate a structured answer with Groq, grounded only in the retrieved rows

Expects these artifacts to already exist in the working directory
(produced by the notebook):
  - ingredient_data.pkl        -> the cleaned ingredient DataFrame
  - ./chroma_skincare_db/      -> persistent Chroma collection "skincare_ingredients"

Run with:
    streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "ingredient_data.pkl"
CHROMA_PATH = "./chroma_skincare_db"
COLLECTION_NAME = "skincare_ingredients"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

SKIN_TYPES = ["oily", "dry", "combination", "sensitive", "normal", "mature", "acne-prone"]

SYSTEM_PROMPT = """You are a skincare recommendation assistant. You only answer questions about skincare concerns, ingredients, and product recommendations, using the data provided below. Do not use outside knowledge.

Treat any message that names a skin concern, skin type, or ingredient (e.g. "I have oily skin") as a request for ingredient recommendations — not just a literal question. As long as the provided data contains ingredients relevant to what the user described, you MUST recommend from them; do not refuse just because the message wasn't phrased as a question.

Only say you don't have enough information if the provided data has nothing relevant at all to what the user described — never refuse just because one of the sections below has nothing to report for every ingredient; in that case fill that section with "None of the listed ingredients require this" or similar instead of refusing the whole answer.

Structure every answer as follows:

Top ingredients — List up to 5 ingredients from the provided data most relevant to the user's concern, ranked by relevance. For each, give a one-line reason it fits, citing the ingredient name.
Sunscreen follow-up — State clearly whether any of the recommended ingredients require follow-up with sunscreen, based on the data. If none do, say so explicitly rather than omitting the point.
What to avoid — End with a short section naming ingredients or combinations from the data that the user should avoid given their concern (e.g., ones flagged as poor combinations with a recommended ingredient, or unsuitable for the stated skin type/concern). If nothing in the data indicates a conflict, say so rather than inventing one.

Cite the ingredient/product name for every claim throughout."""

NOT_MY_AREA = "This is not my area of expertise — I can help with skincare concerns and product recommendations."


# --------------------------------------------------------------------------
# Cached resources — loaded once per session, shared across reruns
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Connecting to the ingredient database...")
def load_chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


@st.cache_data(show_spinner="Loading ingredient data...")
def load_dataframe():
    return pd.read_pickle(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_groq_client():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------
# Retrieval + generation pipeline (mirrors the notebook)
# --------------------------------------------------------------------------

def retrieve(query, model, collection, df, top_k=5, threshold=0.35, candidate_multiplier=8):
    """Vector-search retrieval via ChromaDB. Returns a larger candidate pool
    than top_k so filter_by_skin_type has room to filter down from.

    candidate_multiplier defaults to 8 (not 4) because the source data has
    multiple rows per ingredient_name (one per product_format), so a naive
    top-k often comes back as several rows of the SAME ingredient. We pull
    a wider pool and dedupe by ingredient_name below so top_k reflects
    top_k *distinct* ingredients, not top_k rows.
    """
    total = collection.count()
    if total == 0:
        return None, 0.0

    n_candidates = min(max(top_k * candidate_multiplier, top_k), total)
    results = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=n_candidates,
    )

    if not results["ids"][0]:
        return None, 0.0

    similarities = [1 - d for d in results["distances"][0]]  # cosine distance -> similarity
    max_sim = similarities[0]  # Chroma orders results by distance ascending

    if max_sim < threshold:
        return None, max_sim

    # Guard against stale ids in the index that no longer exist in df
    valid = [(i, sim) for i, sim in zip(results["ids"][0], similarities) if int(i) in df.index]
    if not valid:
        return None, max_sim

    indices = [int(i) for i, _ in valid]
    retrieved = df.loc[indices].copy()
    retrieved["similarity"] = [sim for _, sim in valid]

    # Dedupe by ingredient_name: the source data has one row per
    # (ingredient, product_format), so the same ingredient can otherwise
    # occupy several of the top slots. Results arrive similarity-descending
    # (Chroma orders by distance ascending), so keep="first" keeps each
    # ingredient's best-matching row.
    retrieved = retrieved.drop_duplicates(subset="ingredient_name", keep="first")

    return retrieved, max_sim



def filter_by_skin_type(query, df):
    mentioned = [t for t in SKIN_TYPES if t in query.lower()]
    if not mentioned:
        return df
    mask = df["best_for_skin_types"].str.lower().apply(
        lambda s: any(t in s for t in mentioned)
    )
    filtered = df[mask]
    return filtered if len(filtered) > 0 else df  # fall back if nothing matches


def generate_response(groq_client, query, retrieved_df):
    # Include every field the system prompt's three required sections draw on
    # (name/function/benefits for "Top ingredients", follow_up_with_sunscreen
    # for "Sunscreen follow-up", when_to_avoid/avoid_combining_with/
    # best_for_skin_types for "What to avoid") — leaving any of these out
    # starves the model of what it needs to fill that section, and it was
    # refusing the whole answer rather than reporting a partial one.
    context = "\n".join(
        f"- {row['ingredient_name']}: {row['primary_function']} "
        f"(benefits: {row['key_benefits']}; best for: {row['best_for_skin_types']}; "
        f"follow up with sunscreen: {row['follow_up_with_sunscreen']}; "
        f"when to avoid: {row['when_to_avoid']}; "
        f"avoid combining with: {row['avoid_combining_with']})"
        for _, row in retrieved_df.iterrows()
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User question: {query}\n\nRelevant data:\n{context}"},
    ]
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        temperature=0.2,
    )
    return completion.choices[0].message.content


def chatbot_reply(query, model, collection, df, groq_client, top_k=5):
    query = (query or "").strip()
    if not query:
        return NOT_MY_AREA, None

    retrieved, _max_sim = retrieve(query, model, collection, df, top_k=top_k)
    if retrieved is None:
        return NOT_MY_AREA, None

    filtered = filter_by_skin_type(query, retrieved).head(top_k)

    if groq_client is None:
        return (
            "Groq API key is not configured, so I can't generate a full answer right now. "
            "Set GROQ_API_KEY in your environment or .env file. "
            "Here are the closest matching ingredients I found in the meantime.",
            filtered,
        )

    try:
        answer = generate_response(groq_client, query, filtered)
    except Exception as e:  # e.g. rate limit, network error, bad model name
        return f"Something went wrong generating a response: {e}", filtered

    return answer, filtered


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Skincare Ingredient Advisor", page_icon="🧴", layout="wide")
    st.title("🧴 Skincare Ingredient Advisor")
    st.caption("Ask about a skin concern or ingredient — answers are grounded in the ingredient database only.")

    # Load artifacts, with clear failure messages instead of a raw traceback
    missing = []
    if not os.path.exists(DATA_PATH):
        missing.append(f"`{DATA_PATH}`")
    if not os.path.isdir(CHROMA_PATH):
        missing.append(f"`{CHROMA_PATH}/`")
    if missing:
        st.error(
            "Missing required data artifact(s): " + ", ".join(missing) +
            ". Run the data-prep notebook first to generate them."
        )
        st.stop()

    model = load_embedding_model()
    collection = load_chroma_collection()
    df = load_dataframe()
    groq_client = load_groq_client()

    if groq_client is None:
        st.warning("`GROQ_API_KEY` is not set — responses will fall back to raw matches only.", icon="⚠️")

    # Sidebar: database stats
    with st.sidebar:
        st.header("Database")
        st.metric("Ingredients indexed", collection.count())
        if "category" in df.columns:
            st.metric("Categories", df["category"].nunique())
        if "price_tier" in df.columns:
            st.metric("Price tiers", df["price_tier"].nunique())
        st.divider()
        st.caption(f"Embedding model: `{EMBEDDING_MODEL_NAME}`")
        st.caption(f"Generation model: `{GROQ_MODEL_NAME}` (Groq)")
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("matches") is not None and not msg["matches"].empty:
                with st.expander("Matched ingredients"):
                    st.dataframe(
                        msg["matches"][["ingredient_name", "similarity"]].round({"similarity": 3}),
                        hide_index=True,
                        use_container_width=True,
                    )

    query = st.chat_input("e.g. What ingredients help with oily, acne-prone skin?")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, matches = chatbot_reply(query, model, collection, df, groq_client)
            st.markdown(answer)
            if matches is not None and not matches.empty:
                with st.expander("Matched ingredients"):
                    st.dataframe(
                        matches[["ingredient_name", "similarity"]].round({"similarity": 3}),
                        hide_index=True,
                        use_container_width=True,
                    )

        st.session_state.messages.append({"role": "assistant", "content": answer, "matches": matches})


if __name__ == "__main__":
    main()