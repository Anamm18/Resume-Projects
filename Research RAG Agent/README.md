# Research RAG Agent

A conversational, retrieval-augmented question-answering agent for research papers. The project loads a PDF, splits it into overlapping chunks, embeds those chunks, stores them in a FAISS vector index, and answers natural-language questions about the paper through a conversational retrieval chain, with follow-up questions automatically condensed into standalone queries using the running chat history.

## Overview

The agent is built around a simple pipeline: a research paper in PDF form is loaded and parsed, its text is split into manageable, context-preserving chunks, those chunks are converted into vector embeddings, and the embeddings are indexed for similarity search. When a question is asked, the agent retrieves the most relevant chunks from the index and passes them to a language model, which generates an answer grounded in the retrieved content. Because the chain keeps track of chat history, follow-up questions that depend on earlier context are first rephrased into standalone questions before retrieval, so the agent can hold a coherent multi-turn conversation about the paper rather than treating each question in isolation.

## Tech Stack

The project is written in Python and built on LangChain. Document loading is handled by LangChain's PDF loader, and chunking is done with a recursive character-based text splitter tuned to keep paragraphs intact wherever possible. Embeddings and chat completions are both served through an OpenAI-compatible client, routed via OpenRouter rather than directly through OpenAI, which allows the underlying model to be swapped without changing the application code. FAISS provides the vector store and similarity search. Conversation handling and question condensing are done with LangChain's conversational retrieval chain. Environment configuration is managed with python-dotenv, keeping API keys out of source code.

## How It Works

A PDF research paper is loaded from disk and parsed into page-level documents. These documents are split into chunks of roughly a thousand characters, with a two-hundred-character overlap between consecutive chunks, so that context isn't lost at chunk boundaries. Each chunk is embedded and stored in a FAISS index, which acts as the retriever. A chat model is initialized against OpenRouter using an API key loaded from environment variables. The conversational retrieval chain ties the retriever and the chat model together: on each query, prior chat history and the new question are used to produce a standalone question, relevant chunks are retrieved based on that question, and the chat model composes an answer from those chunks, optionally returning the source documents alongside the answer.

## Project Status

At this stage, the project runs as a notebook-driven proof of concept rather than a packaged application. The PDF path is currently set directly in the code rather than being passed in as a parameter, and the OpenRouter API key is expected to be available as an environment variable named for OpenRouter specifically. The project also currently mixes two dependency declarations: a requirements file listing the core packages, and a pyproject file that does not yet declare any dependencies, so the two should be reconciled before packaging or distributing the project more formally.

## Setup

A Python 3.12 or later environment is expected, matching the version pinned in the project configuration. Dependencies can be installed from the requirements file with pip, or migrated into the pyproject file's dependency list if the project moves toward a packaged, installable structure. An OpenRouter account and API key are required; the key should be placed in a local environment file rather than hardcoded, and it is read at runtime through python-dotenv. Before running the notebook, the path to the target PDF should be updated to point to the research paper to be analyzed on the machine being used, since the current path is specific to one local setup.

## Usage

With the environment configured, the notebook can be run top to bottom: it initializes the chat model and embeddings against OpenRouter, loads and chunks the target PDF, builds the FAISS index from those chunks, and constructs the conversational retrieval chain. Questions are then asked by supplying a query string and the running chat history to the chain, and the generated answer is printed. Starting a new line of questioning with an empty chat history begins a fresh conversational context, while continuing to pass an accumulated history allows the agent to resolve follow-up questions that refer back to earlier turns.

## Known Limitations and Possible Improvements

Several rough edges are worth addressing as the project matures. The hardcoded PDF path should become a configurable input, whether via a command-line argument, a function parameter, or an environment variable, so the notebook can run against any paper without code changes. The document loader currently imported from the community package is being phased out in favor of a standalone integration package, and switching to that package would avoid the deprecation warning and keep the project aligned with LangChain's current recommended imports. The conversational chain being used is itself a somewhat older pattern; migrating toward LangChain's newer agent-based APIs would make it easier to add tools, memory persistence across sessions, and multi-document support later on. Finally, wrapping the pipeline in a small script or a lightweight interface, such as a command-line loop or a simple Streamlit app, would turn this from a one-off notebook into something that's easier to demo and reuse across different papers.
