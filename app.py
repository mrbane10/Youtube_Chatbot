import os
import re
import streamlit as st
import json
import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage, Document
from langchain_groq import ChatGroq
from pytube import Playlist, YouTube
from youtube_transcript_api import YouTubeTranscriptApi as yta
import time
import uuid
import pickle
import hashlib
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(page_title="YouTube Playlist Chatbot", layout="wide")

# Constants
MEMORY_KEY = "chat_history"
WINDOW_MEMORY_K = 2  # Number of previous conversations to include
RETRIEVE_TOP_K = 5  # Number of top chunks to retrieve

# Available LLM models
AVAILABLE_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B Versatile (Default)",
    "llama3-70b-8192": "Llama 3 70B (8K context)",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill Llama 70B",
    "gemma2-9b-it": "Gemma 2 9B IT",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout 17B Instruct",
    "qwen-qwq-32b": "Qwen QWQ 32B",
}

# Use Streamlit's caching mechanisms for file storage
@st.cache_data(show_spinner=False)
def extract_video_id(url):
    """Extract video ID from a YouTube video URL."""
    # Pattern to match various YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shortened
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URL
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # youtu.be format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

@st.cache_data(show_spinner=False)
def is_playlist_url(url):
    """Check if URL is a playlist URL."""
    return 'playlist' in url or 'list=' in url

@st.cache_data(show_spinner=False)
def get_video_info(video_url):
    """Extract info from a single YouTube video URL."""
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return None, None, "Invalid YouTube URL"
            
        yt = YouTube(f"https://youtube.com/watch?v={video_id}")
        return video_id, yt.title, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data(show_spinner=False)
def get_video_ids_from_playlist(playlist_url):
    """Extract video IDs from a YouTube playlist URL."""
    try:
        playlist = Playlist(playlist_url)
        # The playlist.video_urls will give us the full URLs
        # We'll extract the video IDs from them
        video_ids = []
        for url in playlist.video_urls:
            video_id = url.split("v=")[-1].split("&")[0]
            video_ids.append(video_id)
        return video_ids, playlist.title
    except Exception as e:
        st.error(f"Error extracting videos from playlist: {e}")
        return [], ""

@st.cache_data(show_spinner=False)
def fetch_transcripts(video_ids):
    """Download transcripts for all videos in the playlist using youtube_transcript_api."""
    transcripts_dict = {}
    
    progress_text = "Downloading transcripts..."
    progress_bar = st.progress(0.0)
    
    for i, video_id in enumerate(video_ids):
        progress_bar.progress((i + 1) / len(video_ids))
        st.caption(f"{progress_text} ({i+1}/{len(video_ids)})")
            
        try:
            transcript_data = yta.get_transcript(video_id)
            transcript_text = "\n".join([item['text'] for item in transcript_data])
            transcripts_dict[video_id] = transcript_text
            
        except Exception as e:
            st.warning(f"Could not download transcript for video {video_id}: {e}")
    
    # We don't need to save the file to disk in Streamlit Cloud
    # The cache will handle persistence
    return transcripts_dict

@st.cache_resource(show_spinner=False)
def create_embeddings_model():
    """Create embedding model optimized for Streamlit Cloud (CPU-only)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner=False, ttl=3600)
def process_single_video(_video_url):
    """Process a single YouTube video and create vector store."""
    # Get video ID and title
    video_id, video_title, error = get_video_info(_video_url)
    
    if error:
        st.error(f"Error processing video: {error}")
        return None, ""
    
    if not video_id:
        st.error("Could not extract video ID.")
        return None, ""
    
    st.info(f"Processing video: {video_title}")
    
    # Download transcript
    try:
        transcript_data = yta.get_transcript(video_id)
        transcript_text = "\n".join([item['text'] for item in transcript_data])
    except Exception as e:
        st.error(f"Could not download transcript for video: {e}")
        st.info("Make sure the video has captions available.")
        return None, ""
    
    # Create document
    doc = Document(
        page_content=transcript_text,
        metadata={
            "source": f"https://www.youtube.com/watch?v={video_id}",
            "video_id": video_id,
            "title": video_title
        }
    )
    
    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents([doc])
    
    # Create embeddings model (CPU-only for Streamlit Cloud)
    embeddings = create_embeddings_model()
    
    # Create vectorstore
    with st.spinner("Creating vector embeddings..."):
        vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore, video_title

@st.cache_resource(show_spinner=False, ttl=3600)
def process_playlist_documents(_playlist_url):
    """Process the playlist documents and create vector store."""
    # Get video IDs from the playlist
    video_ids, playlist_title = get_video_ids_from_playlist(_playlist_url)
    
    if not video_ids:
        st.error("No videos found in the playlist.")
        return None, ""
    
    st.info(f"Found {len(video_ids)} videos in playlist: {playlist_title}")
    
    # Download transcripts
    transcripts_dict = fetch_transcripts(video_ids)
    
    if not transcripts_dict:
        st.error("Could not download any transcripts.")
        return None, ""
    
    # Convert transcript dictionary to documents for processing
    documents = []
    for video_id, transcript_text in transcripts_dict.items():
        doc = Document(
            page_content=transcript_text,
            metadata={
                "source": f"https://www.youtube.com/watch?v={video_id}",
                "video_id": video_id
            }
        )
        documents.append(doc)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings model (CPU-only for Streamlit Cloud)
    embeddings = create_embeddings_model()
    
    # Create vectorstore
    with st.spinner("Creating vector embeddings..."):
        vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore, playlist_title

def get_conversation_chain(vectorstore, model_name):
    """Create the conversational chain with custom prompt template."""
    # Get API key from secrets or environment
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.session_state.get("groq_api_key", "")
    
    llm = ChatGroq(
        api_key=groq_api_key, 
        model_name=model_name
    )
    
    memory = ConversationBufferWindowMemory(
        memory_key=MEMORY_KEY,
        k=WINDOW_MEMORY_K,
        return_messages=True,
        output_key="answer"
    )
    
    prompt_template = r"""
    You are a helpful assistant that answers questions about YouTube video content.

    You receive two information sources  
    • **Context** - transcript excerpts from the playlist videos (read them first to grasp the topic).  
    • **General knowledge** - your own background expertise, used to fill any gaps.

    ────────────────────────  HOW TO ANSWER  ────────────────────────
    1. **Blend the sources** Merge insights from the transcripts with your knowledge so the reply is complete and coherent.  
    2. **Render every bit of maths in LaTeX**  
    • Spoken phrases → symbols e.g. "sigma squared" → `$\\sigma^{{2}}$`, "x sub i" → `$x_{{i}}$`.  
    • Full equations go in display mode: `$$ … $$`.  
    • Check before sending: every `\left` has a matching `\right`, every `{{` pairs with `}}`, and each `\frac{{…}}{{…}}` has two arguments.  
    3. **Explain step-by-step** Walk through the logic or derivation clearly and in order.  
    4. **No disclaimers** If something is missing from the context, rely on your own knowledge instead of saying "not in context."
    5. **Ensure proper formatting of response**
    ──────────────────────────────────────────────────────────────────

    Context (transcripts):
    {context}

    Conversation history:
    {chat_history}

    Question:
    {question}

    Answer (ensure all equations and symbols are valid LaTeX):
    """

    PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template
    )

    # Create the conversational chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        verbose=True,
        return_source_documents=True
    )
    
    return conversation_chain

def initialize_session_state():
    """Initialize session variables."""
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    if "source_url" not in st.session_state:
        st.session_state.source_url = ""
        
    if "source_title" not in st.session_state:
        st.session_state.source_title = ""
        
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(AVAILABLE_MODELS.keys())[0]  # Default to first model
        
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""
        
    if "source_type" not in st.session_state:
        st.session_state.source_type = "video"  # "video" or "playlist"

def create_new_session(name=None):
    """Create a new session with optional name."""
    session_id = str(uuid.uuid4())
    
    if not name:
        # Default name is based on the session count
        session_count = len(st.session_state.sessions) + 1
        name = f"Session {session_count}"
    
    # Create new conversation chain for the session
    if st.session_state.vectorstore:
        conversation = get_conversation_chain(st.session_state.vectorstore, st.session_state.selected_model)
    else:
        conversation = None
    
    # Store session data
    st.session_state.sessions[session_id] = {
        "name": name,
        "messages": [],
        "conversation": conversation,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": st.session_state.selected_model
    }
    
    # Set as current session
    st.session_state.current_session_id = session_id
    
    return session_id

def get_current_session():
    """Get the current session data."""
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.sessions:
        return st.session_state.sessions[st.session_state.current_session_id]
    return None

def display_messages():
    """Display the conversation messages for the current session."""
    session = get_current_session()
    if session:
        for message in session["messages"]:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):
                    st.write(message.content)

def name_session_from_first_query(query):
    """Generate a session name from the first query."""
    # Extract first few words (up to 5)
    words = query.split()
    if len(words) > 5:
        name = " ".join(words[:5]) + "..."
    else:
        name = query
    
    # Limit length and remove special characters
    name = re.sub(r'[^\w\s]', '', name)
    if len(name) > 30:
        name = name[:30] + "..."
        
    return name

def handle_user_input(user_question):
    """Process user input and update the conversation."""
    session = get_current_session()
    if not session:
        st.error("No active session. Please create a new session first.")
        return
    
    # Check if this is the first message in this session
    is_first_message = len(session["messages"]) == 0
    
    # Add message to session
    session["messages"].append(HumanMessage(content=user_question))
    
    # If this is the first message, update session name
    if is_first_message:
        session_name = name_session_from_first_query(user_question)
        session["name"] = session_name
    
    with st.chat_message("user"):
        st.write(user_question)
    
    with st.spinner(f"Thinking with {AVAILABLE_MODELS.get(session['model'], session['model'])}..."):
        if session["conversation"]:
            # Format chat history for the prompt
            chat_history = ""
            if len(session["messages"]) > 2:  # At least one exchange
                for i in range(max(0, len(session["messages"]) - 4), len(session["messages"]) - 1, 2):
                    if i + 1 < len(session["messages"]):
                        chat_history += f"Human: {session['messages'][i].content}\n"
                        chat_history += f"Assistant: {session['messages'][i+1].content}\n\n"
            
            # Get response from conversation chain
            response = session["conversation"].invoke({
                "question": user_question,
                "chat_history": chat_history
            })
            
            ai_response = response["answer"]
            
            # Show sources if available
            if "source_documents" in response and len(response["source_documents"]) > 0:
                with st.expander("Sources"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1}**: [Video Link]({doc.metadata['source']})")
                        st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
        else:
            ai_response = "Please load a YouTube playlist first."
    
    # Add assistant response to session
    session["messages"].append(AIMessage(content=ai_response))
    
    with st.chat_message("assistant"):
        st.write(ai_response)

def update_session_model(session_id, new_model):
    """Update the model used for a specific session."""
    if session_id in st.session_state.sessions and st.session_state.vectorstore:
        session = st.session_state.sessions[session_id]
        # Update model record
        session["model"] = new_model
        # Create new conversation with the selected model
        session["conversation"] = get_conversation_chain(st.session_state.vectorstore, new_model)
        st.success(f"Session model updated to {AVAILABLE_MODELS.get(new_model, new_model)}")
        return True
    return False

def main():
    """Main application."""
    st.title("YouTube Chatbot")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.subheader("Configuration")
        
        # API Key handling - prefer secrets.toml, but allow manual entry
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            st.warning("GROQ API key not found in environment.")
            groq_key_input = st.text_input("Enter your Groq API key:", type="password", value=st.session_state.groq_api_key)
            if groq_key_input:
                st.session_state.groq_api_key = groq_key_input
                
            st.info("For Streamlit Cloud deployment, set this as a secret in your app settings.")
        else:
            st.success("GROQ API key found in environment.")
        
        # Model selection
        st.subheader("Model Selection")
        selected_model = st.selectbox(
            "Choose LLM Model",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS.get(x, x),
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model)
        )
        
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            # Update current session's model if one exists
            if st.session_state.current_session_id:
                update_session_model(st.session_state.current_session_id, selected_model)
        
        # YouTube source input
        st.subheader("YouTube Source")
        
        # Let user choose between single video or playlist
        source_type = st.radio(
            "Select source type:",
            options=["Single Video", "Playlist"],
            index=0 if st.session_state.source_type == "video" else 1
        )
        
        st.session_state.source_type = "video" if source_type == "Single Video" else "playlist"
        
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            help="Enter a YouTube video URL or a playlist URL"
        )
        
        load_button_label = "Load Video" if st.session_state.source_type == "video" else "Load Playlist"
        if st.button(load_button_label):
            if not youtube_url:
                st.error(f"Please enter a YouTube {st.session_state.source_type} URL")
            elif not (os.environ.get("GROQ_API_KEY") or st.session_state.groq_api_key):
                st.error("Please provide a Groq API key")
            else:
                # Check if URL is valid for the selected type
                if st.session_state.source_type == "playlist" and not is_playlist_url(youtube_url):
                    st.error("The URL doesn't appear to be a playlist. Please enter a playlist URL or switch to 'Single Video' mode.")
                elif st.session_state.source_type == "video" and is_playlist_url(youtube_url):
                    st.error("The URL appears to be a playlist. Please enter a single video URL or switch to 'Playlist' mode.")
                else:
                    with st.spinner(f"Processing {st.session_state.source_type}... This may take a while"):
                        if st.session_state.source_type == "video":
                            # Process single video
                            vectorstore, video_title = process_single_video(youtube_url)
                            source_title = video_title
                        else:
                            # Process playlist
                            vectorstore, playlist_title = process_playlist_documents(youtube_url)
                            source_title = playlist_title
                        
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.source_title = source_title
                            st.session_state.source_url = youtube_url
                            
                            # Create initial session
                            if not st.session_state.current_session_id:
                                create_new_session(f"New {source_title} Session")
                            else:
                                # Update conversation chain for all sessions
                                for session_id in st.session_state.sessions:
                                    model = st.session_state.sessions[session_id]["model"]
                                    st.session_state.sessions[session_id]["conversation"] = get_conversation_chain(vectorstore, model)
                            
                            st.success(f"{source_type} processed successfully! You can now ask questions about it.")
        
        if st.session_state.source_url:
            source_type_display = "Video" if st.session_state.source_type == "video" else "Playlist"
            st.success(f"Currently loaded {source_type_display}: {st.session_state.source_title or st.session_state.source_url}")
            
            # Session management
            st.subheader("Session Management")
            
            # Create new session button
            if st.button("New Session"):
                create_new_session()
                st.rerun()
            
            # Session selector
            if st.session_state.sessions:
                session_options = {s_id: f"{session['name']} ({AVAILABLE_MODELS.get(session['model'], session['model']).split(' ')[0]})" 
                                  for s_id, session in st.session_state.sessions.items()}
                selected_session = st.selectbox(
                    "Select Session", 
                    options=list(session_options.keys()),
                    format_func=lambda x: session_options[x],
                    index=list(session_options.keys()).index(st.session_state.current_session_id) if st.session_state.current_session_id in session_options else 0
                )
                
                if selected_session != st.session_state.current_session_id:
                    st.session_state.current_session_id = selected_session
                    st.rerun()
            
            # Session actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Current Session"):
                    session = get_current_session()
                    if session:
                        session["messages"] = []
                        session["conversation"] = get_conversation_chain(st.session_state.vectorstore, session["model"])
                        st.rerun()
            
            with col2:
                if st.button("Delete Current Session"):
                    if st.session_state.current_session_id:
                        del st.session_state.sessions[st.session_state.current_session_id]
                        if st.session_state.sessions:
                            st.session_state.current_session_id = next(iter(st.session_state.sessions))
                        else:
                            st.session_state.current_session_id = None
                        st.rerun()
            
            # Session info
            session = get_current_session()
            if session:
                st.text(f"Current Session: {session['name']}")
                st.text(f"Model: {AVAILABLE_MODELS.get(session['model'], session['model'])}")
                st.text(f"Created: {session['created_at']}")
                st.text(f"Messages: {len(session['messages'])}")
                
                # Individual session model override
                st.subheader("Session Model")
                session_model = st.selectbox(
                    "Change Session Model",
                    options=list(AVAILABLE_MODELS.keys()),
                    format_func=lambda x: AVAILABLE_MODELS.get(x, x),
                    index=list(AVAILABLE_MODELS.keys()).index(session["model"]),
                    key="session_model_override"
                )
                
                if st.button("Update Session Model"):
                    if session_model != session["model"]:
                        update_session_model(st.session_state.current_session_id, session_model)
                        st.rerun()
    
    # Main chat interface
    if st.session_state.current_session_id:
        display_messages()
        
        # User input
        source_type = "video" if st.session_state.source_type == "video" else "playlist"
        user_question = st.chat_input(f"Ask about the YouTube {source_type}:")
        if user_question:
            handle_user_input(user_question)
    else:
        if st.session_state.source_url:
            st.info("Please create a new session to start chatting.")
        else:
            st.info("Please load a YouTube video or playlist to start.")

if __name__ == "__main__":
    main()
