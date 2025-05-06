import google.generativeai as genai
import os

class LLMClient:
    def __init__(self, model_name="gemini-1.5-pro-latest", api_key: str = None, system_instruction: str = None):
        """
        Initializes the LLM client using the Google Generative AI SDK.

        Args:
            model_name (str): The name of the Google Gemini model to use.
            api_key (str, optional): Your Google API key. If None, uses GOOGLE_API_KEY env var.
            system_instruction (str, optional): Default system instruction for the model.
        """
        self.model_name = model_name
        
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ValueError("Google API key must be provided or set as GOOGLE_API_KEY environment variable.")
        genai.configure(api_key=resolved_api_key)

        try:
            self.model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_instruction
            )
        except Exception as e:
            print(f"Error initializing GenerativeModel '{self.model_name}': {e}")
            print("Please ensure API key is valid, model name correct, and system_instruction (if any) is valid.")
            raise

    def chat(self, prompt: str, stream: bool = False, history: list = None):
        """
        Sends a prompt to the Gemini LLM and gets a response, potentially using chat history.

        Args:
            prompt (str): The user's prompt.
            stream (bool): Whether to stream the response.
            history (list, optional): A list of previous messages for the chat session,
                                      formatted as {'role': 'user/model', 'parts': [{'text': '...'}]}.
                                      If None, a new chat session is started.

        Returns:
            If stream is False: google.generativeai.types.GenerateContentResponse
            If stream is True: generator yielding google.generativeai.types.GenerateContentResponse chunks.
        """
        try:
            chat_session = self.model.start_chat(history=history or [])
            response = chat_session.send_message(prompt, stream=stream)
            return response
        except Exception as e:
            print(f"Error communicating with Google Gemini API (model: '{self.model_name}'): {e}")
            # Return a consistent error structure or re-raise based on desired error handling
            return {"error": str(e), "status_code": getattr(e, 'code', None) or getattr(e, 'status_code', 'Unknown')}
