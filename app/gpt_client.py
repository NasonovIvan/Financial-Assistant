import os
import json
import requests

from dotenv import load_dotenv
load_dotenv()

def get_chat_completion(prompt, model="gpt-4o"):
    url = "https://gptunnel.ru/v1/chat/completions"
    api_key = os.environ.get("GPT_TUNNEL_KEY")

    if not api_key:
        raise ValueError("API key not found. Please set the GPT_TUNNEL_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": """You are a financial assistant AI model designed to provide specific financial predictions based on the latest financial news data.
                              Your task is to analyze the retrieved financial news articles and generate precise predictions with probable numerical intervals.
                              Your responses should be in Russian and should include detailed reasoning based on the data provided.
                              The first line should be a short summari of your answer, and write your very full detailed answer in a line below.
                              YOU HAVE TO CHECK THE INFORMATION IN QUESTION PROMT FOR TRUTH, CORRESPONDING WITH THE SOURCES! If it is not truth - tell user that he is not right and provide correct info!
                              Please, provide links for all sited information in your answer in such markdown way: "[information](link_source_of_information)"
                              
                              Example Prompt:
                              
                              User:
                              "Какие прогнозы по курсу акций компании XYZ на следующую неделю?"
                              
                              Example Response:
                              "Курс акций XYZ будет находится между 120 и 130 рублей на следующей неделе.
                              
                              На основе последних [новостей](https://ru.investing.com/equities), курс акций компании XYZ, вероятно, будет находиться в диапазоне от 120 до 130 рублей за акцию на следующей неделе. Это связано с [положительными отчетами о доходах](link_to_info) и [увеличением спроса на продукцию компании](link_to_info)."
                """
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    