import streamlit as st
from streamlit_chat import message
from chatbot_code import TravelServiceChatbot
import os

# Page config
st.set_page_config(
    page_title="Travel Service Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7f9;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

# Sidebar
with st.sidebar:
    st.title("✈️ Travel Assistant Settings")
    st.markdown("---")
    
    # API Key handling
    try:
        # Try to get API key from secrets
        gemini_api_key = st.secrets['GEMINI_API_KEY']
        if st.session_state.chatbot is None:
            st.session_state.chatbot = TravelServiceChatbot(gemini_api_key)
            st.success("Chatbot initialized successfully!")
    except:
        # If no secret is found, show input field
        gemini_api_key = st.text_input("Enter Gemini API Key:", type="password")
        if gemini_api_key:
            if st.session_state.chatbot is None:
                st.session_state.chatbot = TravelServiceChatbot(gemini_api_key)
                st.success("Chatbot initialized successfully!")
    
    # Knowledge Base Upload
    st.subheader("Knowledge Base")
    uploaded_file = st.file_uploader("Upload Knowledge Base", type=['txt'])
    if uploaded_file and st.session_state.chatbot:
        content = uploaded_file.getvalue().decode()
        num_paragraphs = st.session_state.chatbot.load_knowledge_base(content)
        st.success(f"Loaded {num_paragraphs} paragraphs into knowledge base")
    
    # Admin controls
    st.subheader("Admin Controls")
    new_knowledge = st.text_area("Add New Knowledge", height=150)
    if st.button("Add Knowledge"):
        if st.session_state.chatbot and new_knowledge.strip():
            doc_id = st.session_state.chatbot.add_knowledge(new_knowledge)
            st.success(f"Added new knowledge with ID: {doc_id}")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# Main chat interface
st.title("Travel Service Assistant")
st.markdown("How can I help you with your travel plans today? 🌎")

# Display chat messages
for i, (role, content) in enumerate(st.session_state.messages):
    if role == "User":
        message(content, is_user=True, key=f"msg_{i}_user")
    else:
        message(content, key=f"msg_{i}_assistant")

# Chat input
if st.session_state.chatbot:
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        st.session_state.messages.append(("User", user_input))
        
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.chat(user_input)
            st.session_state.messages.append(("Assistant", response))
            st.experimental_rerun()
else:
    st.warning("Please enter your Gemini API key in the sidebar to start chatting.")