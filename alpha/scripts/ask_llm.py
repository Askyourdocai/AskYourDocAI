""" This module is used to interact with the  LLM using the langchain library. """

import os

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

# Ensure environmental variables are loaded
load_dotenv()


# pylint: disable=R0903
class AskGPT3:
    """This class is used to interact with the GPT-3.5-turbo model using the langchain library.

    Examples:
        >>> ask_gpt3 = AskGPT3()
        >>> question = "What is the capital of France?"
        >>> retrieved_documents = [["Paris is the capital of France."]]
        >>> response = ask_gpt3.ask_gpt3(question, retrieved_documents)
        >>> print(response)
    """

    def __init__(self):
        """Initializes the AskGPT3 object."""
        # Load the OpenAI API Key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        # Define the template for conversation
        template = """The following is a text extracting conversation between a user and an AI.
                        The AI is a text extractor and extracts the answer for the question from the user given context and also from the previous coversational history.
                        The AI keeps answers concise and to the point.
                        If the AI does not know the answer to a question, it truthfully says it does not know.

                    Current conversation:
                    {history} \n
                    question context: {input}
                    AI Assistant:"""
        # Initialize the PromptTemplate
        self.prompt = PromptTemplate(
            input_variables=["history", "input"], template=template
        )
        # Initialize the ChatOpenAI with the GPT-3.5-turbo model
        chat = ChatOpenAI(temperature=0, api_key=openai_api_key, model="gpt-3.5-turbo")
        # Initialize the ConversationChain with the specified memory window
        self.conversation = ConversationChain(
            llm=chat,
            prompt=self.prompt,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=4),
        )

    def ask_gpt3(self, question, retrieved_documents):
        """Asks the GPT-3.5-turbo model a question and returns the response.

        Args:
            question (str): The question to ask the model.
            retrieved_documents (list): A list of retrieved documents.

        Returns:
            str: The response from the model.
        """
        context = " ".join(retrieved_documents[0])
        # Inject context and query into the conversation
        response = self.conversation.predict(input=f"{context} \n question: {question}")
        return response
