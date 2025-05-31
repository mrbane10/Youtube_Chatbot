# YouTube Playlist Chatbot

**YouTube Playlist Chatbot** is a Streamlit app that lets you chat with any YouTube video or playlist.  
It fetches transcripts, builds semantic search, and uses Groq LLMs to answer your questions—rendering math in LaTeX, showing sources, and supporting multiple sessions and models.

---

## Features

- **Chat with any YouTube video or playlist:** Ask questions with answers grounded in transcripts.
- **Multiple Groq LLMs:** Choose from several models (Llama 3, Llama 4, DeepSeek, Gemma, Qwen, etc).
- **LaTeX math rendering:** Spoken math is converted to proper LaTeX.
- **Full playlist support:** Ask questions across all videos in a playlist.
- **Sessions & model switching:** Manage different conversations and swap models per session.
- **Transparent sources:** See which transcript segment was used for each answer.
- **Streamlit-friendly:** CPU-only embeddings, runs anywhere.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit langchain langchain-groq pytube youtube-transcript-api sentence-transformers faiss-cpu torch
```


### 2. Set Your Groq API Key
Option 1: Environment variable
```
export GROQ_API_KEY=sk-xxxx
```
Option 2: Streamlit secret
In `.streamlit/secrets.toml`:

toml
Copy
Edit
GROQ_API_KEY = "sk-xxxx"
3. Run the App
bash
```
streamlit run your_script.py
```

## Usage
- Enter a YouTube video or playlist URL in the sidebar.

- Load the video or playlist.

- Ask questions about the content.

- (Optional) Manage multiple sessions or change LLM models.

- Proxy support: Add `WEBSHARE_PROXY_USERNAME` and `WEBSHARE_PROXY_PASSWORD` to Streamlit secrets if you need a proxy to download the transcripts as youtube blocks the ip for multiple requests.`

- Custom models: Edit the `AVAILABLE_MODELS` dictionary in the code to add more LLMs.

## Security
Only transcripts are fetched—no video/audio is downloaded.

Groq API key is local or in Streamlit secrets.

License
MIT
