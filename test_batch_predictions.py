import pandas as pd
import os
# from OpenAIAPI import Summarizer
from llmsapi import llmsAPI
import logging
import time
from icecream import ic
from config import *
from tqdm import tqdm


{
    "custom_id": "request-1", 
    "method": "POST", 
    "url": "/v1/chat/completions", 
    "body": 
        {
            "model": "gpt-3.5-turbo-0125", 
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello world!"}],
            "temperature": 0,
            "max_tokens": 1000,
            "top_p":1,
            "seed":42
            }
        }
