""" Streamlit UI for the Ask Your Doc AI bot. """

import base64
import tempfile

from ask_llm import AskGPT3
from retrieve_relevant_docs import RetrieveRelevantDocsBase
from retrieve_relevant_docs import RetrieveRelevantDocsChromaDB
from retrieve_relevant_docs import RetrieveRelevantDocsPinecone
import streamlit as st


def get_base64(bin_file):
    """Get the base64 encoding of a binary file.

    Args:
        bin_file (str): The path to the binary file.

    Returns:
        str: The base64 encoding of the binary file.
    """
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    """Set the background of the page to a PNG image.

    Args:
        png_file (str): The path to the PNG file.

    """
    bin_str = get_base64(png_file)
    page_bg_img = (
        """
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    """
        % bin_str
    )
    st.markdown(page_bg_img, unsafe_allow_html=True)


# set_background('image_path.png')


# Streamlit UI setup
st.title("Ask Your Doc AI 🤖")

# File Uploader and Processing at the top after the title
pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if pdf_file is not None:
    with st.spinner("Uploading file..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.getvalue())
            pdf_path = tmpfile.name

# LLM Model Selection in the sidebar
llm_model = st.sidebar.selectbox(
    "Which LLM do you want to use?",
    ("gpt-3.5-turbo", "mistral"),
    index=0,  # Default to first option
)

# Initialize the AskGPT3 class only once using Streamlit's session state
if "ask_llm" not in st.session_state or llm_model != st.session_state["last_llm_model"]:
    if llm_model == "gpt-3.5-turbo":
        st.session_state.ask_llm = AskGPT3()
    # Assuming you have a different class or method for 'mistral'
    # elif llm_model == "mistral":
    #     st.session_state.ask_llm = AskMistral()  # Example, replace with actual initialization
    st.session_state["last_llm_model"] = llm_model

# Refer to the 'ask_llm' instance from the session state
ask_llm = st.session_state.ask_llm


# RAG VectorDB Selection in the sidebar
rag_vectordb = st.sidebar.selectbox(
    "Which RAG VectorDB do you want to use?", ("ChromaDB", "Pinecone")
)

# Initialize relevant document retrieval system based on the RAG VectorDB selection
if pdf_file is not None:
    if rag_vectordb == "Pinecone":
        rrds: RetrieveRelevantDocsBase = RetrieveRelevantDocsPinecone(pdf_path)
    else:
        rrds = RetrieveRelevantDocsChromaDB(pdf_path)
    st.success("File uploaded successfully!")

# Initialize session state for chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Function to handle new messages
def on_new_message_entered():
    """Handle new messages entered by the user."""
    user_query = st.session_state.new_message  # Get the current message from the input
    if user_query:  # Check if the message is not empty
        # Process the user query here and generate a response
        retrieved_documents = (
            rrds.get_relevant_docs(user_query, n_results=2)
            if pdf_file is not None
            else []
        )
        # Inject context and query into the conversation
        response = ask_llm.ask_gpt3(user_query, retrieved_documents)

        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("ai", response))
        # Clear the input box for the next message
        st.session_state.new_message = ""


# Function to apply styles directly (alternative to using a local CSS file)
def apply_styles():
    """Apply custom styles to the chat messages."""
    st.markdown(
        """
        <style>
        .user-message {
            background-color: rgba(0, 123, 255, 0.1); /* Transparent blue */
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0px;
        }
        .ai-message {
            background-color: rgba(40, 167, 69, 0.1); /* Transparent green */
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


# Apply the styles
apply_styles()


# Display chat history in a WhatsApp-like manner within a container
chat_container = st.container()
chat_container.write("## Chat")
with chat_container:
    for message_type, message in st.session_state.chat_history:
        if message_type == "user":
            col1, col2 = st.columns([1, 3])
            col2.markdown(
                f'<div class="user-message">😁: {message}</div>', unsafe_allow_html=True
            )
        else:  # AI messages
            col1, col2 = st.columns([3, 1])
            col1.markdown(
                f'<div class="ai-message">🤖: {message}</div>', unsafe_allow_html=True
            )

# Input for new message below the chat container
new_message = st.text_input(
    "",
    placeholder="Type your message here and press enter:",
    key="new_message",
    on_change=on_new_message_entered,
)
