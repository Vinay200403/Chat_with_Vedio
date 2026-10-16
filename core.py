import os
import re
import time

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Create a .env file next to app.py with: "
        "GEMINI_API_KEY=your_key_here"
    )

# Models are configurable through .env so nothing is hardcoded.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "")  # optional

ai_client = genai.Client(api_key=api_key)

# Persistent ChromaDB client so vectors are saved locally
client = chromadb.PersistentClient(path="./chroma_db")
embedding_func = embedding_functions.DefaultEmbeddingFunction()


def extract_video_id(url: str) -> str:
    """Extracts the 11-character YouTube video ID from various URL formats."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL provided.")
    return match.group(1)


def get_raw_transcript(video_id: str):
    """Fetch transcript using youtube-transcript-api 1.x.

    Tries preferred languages first, then falls back to any available
    transcript (manual or auto-generated).
    """
    api = YouTubeTranscriptApi()
    try:
        return api.fetch(video_id, languages=["en", "en-IN", "hi"]).to_raw_data()
    except Exception:
        transcript_list = api.list(video_id)
        first = next(iter(transcript_list))
        return first.fetch().to_raw_data()


def fetch_and_chunk_transcript(video_id: str, max_chunk_words: int = 150):
    """Fetch the transcript and chunk it, keeping each chunk's start time."""
    try:
        transcript = get_raw_transcript(video_id)
    except Exception as e:
        raise ValueError(f"Could not fetch transcript: {e}")

    chunks = []
    current_text = []
    word_count = 0
    chunk_start_time = 0.0

    for entry in transcript:
        if not current_text:
            chunk_start_time = entry["start"]

        current_text.append(entry["text"])
        word_count += len(entry["text"].split())

        if word_count >= max_chunk_words:
            chunks.append(
                {"text": " ".join(current_text), "start_seconds": int(chunk_start_time)}
            )
            current_text = []
            word_count = 0

    if current_text:
        chunks.append(
            {"text": " ".join(current_text), "start_seconds": int(chunk_start_time)}
        )

    return chunks


def index_video_transcript(video_id: str):
    """Index timestamped video chunks into a ChromaDB collection."""
    collection_name = f"video_{video_id}"

    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_func
    )

    # Already indexed: skip
    if collection.count() > 0:
        return collection

    try:
        chunks = fetch_and_chunk_transcript(video_id)
        if not chunks:
            raise ValueError("The transcript was empty.")

        documents = [c["text"] for c in chunks]
        metadatas = [{"start_seconds": c["start_seconds"]} for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    except Exception:
        # Don't leave an empty collection behind after a failed attempt
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        raise

    return collection


def generate_with_retry(prompt: str, retries: int = 4):
    """Call Gemini, retrying temporary errors (503/429) and trying a fallback model."""
    models = [MODEL_NAME] + ([FALLBACK_MODEL] if FALLBACK_MODEL else [])
    last_error = None
    for model in models:
        for attempt in range(retries):
            try:
                return ai_client.models.generate_content(model=model, contents=prompt)
            except Exception as e:
                last_error = e
                msg = str(e)
                if "503" in msg or "429" in msg or "UNAVAILABLE" in msg:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s
                    continue
                raise
    raise last_error


def _format_history(history, max_messages: int = 6) -> str:
    """Turn the last few chat messages into plain text for the prompt."""
    if not history:
        return ""
    lines = []
    for m in history[-max_messages:]:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


def _standalone_question(user_query: str, history) -> str:
    """Rewrite a follow-up ("what about the second one?") into a standalone
    question so the vector search finds the right transcript chunks.
    Falls back to the original question if anything goes wrong."""
    if not history:
        return user_query
    try:
        prompt = f"""Rewrite the user's latest message as a single standalone question
that can be understood without the conversation. Keep the original language.
If it is already standalone, return it unchanged. Return ONLY the question.

Conversation:
{_format_history(history)}

Latest message: {user_query}"""
        rewritten = generate_with_retry(prompt, retries=2).text.strip()
        return rewritten or user_query
    except Exception:
        return user_query


def query_video(collection, user_query: str, history=None, n_results: int = 4):
    """Retrieve relevant chunks and generate a grounded answer with Gemini.

    `history` is the earlier chat as a list of {"role", "content"} dicts, so
    follow-up questions work like a normal chat.
    """
    search_query = _standalone_question(user_query, history)
    results = collection.query(query_texts=[search_query], n_results=n_results)

    context_blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_blocks.append(
            f"Transcript Segment (Timestamp: {meta['start_seconds']}s): {doc}"
        )

    combined_context = "\n\n".join(context_blocks)
    history_text = _format_history(history)
    history_block = f"Conversation so far:\n{history_text}\n\n" if history_text else ""

    prompt = f"""
    You are a helpful assistant chatting with a user about a video.
    Answer using the transcript context below. Use the conversation so far to
    understand follow-up questions. If the context does not contain the answer,
    say so instead of guessing.

    CRITICAL INSTRUCTION: You MUST cite the source of your information by appending
    the exact timestamp provided in the context in brackets, like this: [120s].
    Do not invent timestamps. Only use the timestamps provided in the context blocks below.

    {history_block}Context:
    {combined_context}

    Question: {user_query}
    """

    response = generate_with_retry(prompt)

    return {"llm_answer": response.text, "raw_results": results}
