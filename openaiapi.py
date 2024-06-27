import os
# import openai
from openai import OpenAI
import logging
from icecream import ic

# Configure the logger
logging.basicConfig(filename='summarizer.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

client = OpenAI(api_key = os.environ.get('OPENAI_KEY'))

class OpenAIAPI:
    def __init__(self, prompt):
        self.prompt = prompt

    def get_completion(self, model="gpt-3.5-turbo-0125"):
        try:
            # To prevent maintaining context, we keep only the necessary message
            response = client.chat.completions.create(
                # response_format={ "type": "json_object" },
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in predicting the number of necessary documents to attend to the user's needs."},
                    {"role": "user", "content": self.prompt}],
                temperature=0,  # this is the degree of randomness of the model's output
                max_tokens = 10, ##TODO: change this to 10
                top_p=1,
                seed=42
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"An error occurred during the OpenAI API call: {e}")
            # You can add any custom error handling or logging here
            return None