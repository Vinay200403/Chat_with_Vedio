# 🎥 Talk With Video (VideoRAG)

Chat with any YouTube video. The app fetches the transcript, indexes timestamped chunks in a local ChromaDB, answers your questions with Gemini, and lets you jump the embedded player to the cited moments.

## Setup (Windows)

"""powershell
python -m venv .myvenv
.\.myvenv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then open .env and paste your Gemini API key
streamlit run app.py
"""

Opens at http://localhost:8501.

## Configuration (.env)

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | yes | Your Google AI Studio key |
| `GEMINI_MODEL` | no | Main model (default `gemini-3.8-flash`) |
| `GEMINI_FALLBACK_MODEL` | no | Used if the main model keeps returning 503/429 |

## How to use

1. Paste a YouTube URL in the sidebar and click **Process & Index Video**.
2. Chat in the right panel. The chat stays open, so keep asking follow-up questions; it remembers the conversation.
3. Click a **▶ MM:SS** button under any answer to seek the player to that moment.
4. Use **Clear chat** in the sidebar to start over. Indexing a new video also starts a fresh chat.

## Troubleshooting

- **`no element found: line 1, column 0`**: outdated `youtube-transcript-api`. Run `pip install -U youtube-transcript-api` inside the venv.
- **`The read operation timed out`**: usually the first-run download of the embedding model (~80 MB). Retry, or switch network. If it was interrupted, delete `C:\Users\<you>\.cache\chroma\onnx_models`.
- **`404 NOT_FOUND` model error**: set `GEMINI_MODEL` in `.env` to a model your key can use.
- **`503 UNAVAILABLE`**: Google's model is overloaded. The app retries automatically; set `GEMINI_FALLBACK_MODEL` for extra resilience.
- **Transcript blocked**: YouTube may block some IPs/VPNs. Try another network.

## Structure

```
core.py            transcript fetch, chunking, ChromaDB, Gemini calls
app.py             Streamlit UI
chroma_db/         auto-generated local vector store
.env               your secrets (not committed)
```
