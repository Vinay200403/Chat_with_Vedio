import re

import streamlit as st

from core import extract_video_id, index_video_transcript, query_video

st.set_page_config(page_title="Talk With Video", layout="wide")

st.title("🎥 Talk With Video")
st.caption(
    "Chat with any YouTube video. Answers are grounded in the transcript, "
    "with clickable timestamps."
)

# ---------- session state ----------
st.session_state.setdefault("messages", [])  # the chat history
st.session_state.setdefault("seek_time", 0)


def set_seek(sec: int):
    st.session_state["seek_time"] = sec


def clear_chat():
    st.session_state["messages"] = []


def render_message(msg: dict, idx: int):
    """Draw one chat bubble, plus 'Jump to' buttons for cited timestamps."""
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and not msg.get("error"):
            times = sorted(set(int(t) for t in re.findall(r"\[(\d+)s\]", msg["content"])))
            if times:
                cols = st.columns(min(len(times), 5))
                for i, sec in enumerate(times[:5]):
                    mins, secs = divmod(sec, 60)
                    with cols[i]:
                        st.button(
                            f"▶ {mins:02d}:{secs:02d}",
                            key=f"jump_{idx}_{sec}",
                            on_click=set_seek,
                            args=(sec,),
                        )
                        


# ---------- sidebar ----------
with st.sidebar:
    st.header("Video")
    video_url = st.text_input("Enter YouTube Video URL:")
    index_btn = st.button("Process & Index Video", use_container_width=True)

    if "video_id" in st.session_state:
        st.caption(f"Loaded: `{st.session_state['video_id']}`")

    if st.session_state["messages"]:
        st.button("🗑️ Clear chat", on_click=clear_chat, use_container_width=True)

# Index when the button is clicked
if video_url and index_btn:
    try:
        with st.spinner("Extracting captions and indexing into ChromaDB..."):
            vid_id = extract_video_id(video_url)
            collection = index_video_transcript(vid_id)
        st.session_state["video_id"] = vid_id
        st.session_state["collection"] = collection
        st.session_state["messages"] = []  # new video = fresh chat
        st.session_state["seek_time"] = 0
        st.sidebar.success("Video indexed! Start chatting.")
    except Exception as e:
        st.error(f"Error: {e}")

# ---------- main layout ----------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Video Player")
    if "video_id" in st.session_state:
        st.video(
            f"https://www.youtube.com/watch?v={st.session_state['video_id']}",
            start_time=st.session_state["seek_time"],
        )
    else:
        st.info("Provide a video URL in the sidebar to load the player.")

with col2:
    st.subheader("Chat")
    ready = "collection" in st.session_state

    chat_box = st.container(height=520, border=True)
    prompt = st.chat_input(
        "Ask anything about the video..." if ready else "Index a video first",
        disabled=not ready,
    )

    with chat_box:
        if not ready:
            st.info("Index a video in the sidebar to start chatting.")
        elif not st.session_state["messages"] and not prompt:
            st.caption("Ask a question to get started. You can keep asking follow-ups.")

        # Earlier messages
        for i, m in enumerate(st.session_state["messages"]):
            render_message(m, i)

        # New message: answer it, then store both sides of the exchange
        if ready and prompt:
            # History for the model = earlier successful messages only
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"]
                if not m.get("error")
            ]

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = query_video(
                            st.session_state["collection"], prompt, history=history
                        )
                        answer, is_error = result["llm_answer"], False
                    except Exception as e:
                        answer, is_error = f"⚠️ Could not get an answer: {e}", True

            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "error": is_error}
            )
            # Rerun so the new answer is drawn as a normal message with its
            # jump buttons, and the chat box is ready for the next question.
            st.rerun()
