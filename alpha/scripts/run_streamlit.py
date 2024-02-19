"""This module sets up a Streamlit web interface for interacting with LLMs."""

import tempfile
import time

import openai
from openai import OpenAI
from retrieve_relevant_docs import RetrieveRelevantDocs
import streamlit as st
from system_prompt import SystemPrompt


# Streamlit UI setup
st.title("Ask Your Doc AI")

# LLM Model Selection
llm_model = st.selectbox(
    "Which LLM do you want to use?",
    ("gpt-3.5-turbo", "mistral"),
    index=0,  # Default to first option
)
# Check if the selected LLM is a GPT model
if "gpt" in llm_model:
    # Input for API Key
    openai_api_key = st.text_input("Enter your OpenAI API key:")
    openai.api_key = openai_api_key


# File Uploader
pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if pdf_file is not None:
    with st.spinner("Uploading file..."):
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.getvalue())
            pdf_path = tmpfile.name

    rrds = RetrieveRelevantDocs(pdf_path)

    st.write("File uploaded successfully!")

    # Query Input
    query = st.text_input("Ask anything about your doc:")
    if st.button("Submit"):
        # Display a progress bar
        progress_bar = st.progress(0)

        # Example: Update the progress bar with a loop
        for percent_complete in range(100):
            time.sleep(0.1)  # Simulate a task
            progress_bar.progress(percent_complete + 1)

        retrieved_documents = rrds.get_relevant_docs(query, n_results=5)
        # pylint: disable=invalid-name
        context = " ".join(retrieved_documents[0])

        system_prompt = SystemPrompt(llm_model).get_system_prompt()

        # Replace placeholders
        system_prompt[1]["content"] = system_prompt[1]["content"].replace(
            "{{context}}", context
        )
        system_prompt[1]["content"] = system_prompt[1]["content"].replace(
            "{{question}}", query
        )

        client = OpenAI(api_key=openai_api_key)
        completion = client.chat.completions.create(
            messages=system_prompt,
            model=llm_model,
            max_tokens=150,
        )

        # Set the progress bar to 100% at the end
        progress_bar.progress(100)

        st.write("Answer: ", completion.choices[0].message.content)
