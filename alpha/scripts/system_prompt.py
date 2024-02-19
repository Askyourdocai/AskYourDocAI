"""This script is used to generate prompts for respective LLMs with placeholders."""


class SystemPrompt:
    """This class generates prompts for respective LLMs with placeholders."""

    def __init__(self, llm_model):
        """Initialize the class with the LLM model."""
        self.llm_model = llm_model

    def get_system_prompt(self):
        """Get the system prompt with placeholders."""
        if self.llm_model == "gpt-3.5-turbo":
            return self.get_system_prompt_gpt_3_5_turbo()
        if self.llm_model == "mistral":
            return self.get_system_prompt_mistral()
        return "Invalid LLM model."

    def get_system_prompt_gpt_3_5_turbo(self):
        """Get the system prompt for gpt-3.5-turbo LLM with specific formatting."""
        return [
            {
                "role": "system",
                "content": "You are a text extractor and extract the answer for the question from the given context."
                "Keep answers concise and to the point.",
            },
            {
                "role": "user",
                "content": "context: {{context}}\nquestion: {{question}}\nanswer:",
            },
        ]

    def get_system_prompt_mistral(self):
        """Get the system prompt for the mistral LLM with specific formatting."""
        return """[INST] <<SYS>>\nYou are a text extractor and extract the answer for the question from the given
                  context. Keep answers concise and to the point. Context: {{context}}\nQuestion: {{question}} 
                  [/INST]"""
