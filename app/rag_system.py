import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timezone
import re
import emoji
import requests
# from bs4 import BeautifulSoup

class RAGSystem:
    def __init__(self, data_file='data/telegram_messages.json', index_file='data/faiss_index.idx'):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')
        self.documents = []
        self.embeddings = None
        self.data_file = data_file
        self.index_file = index_file
        self.tfidf_vectorizer = TfidfVectorizer()
        self.load_or_create_index()

    def load_or_create_index(self):
        if os.path.exists(self.data_file) and os.path.exists(self.index_file):
            self.load_documents()
            self.load_index()
        else:
            print("Index or data file not found. Please update the database.")

    def load_documents(self):
        with open(self.data_file, 'r', encoding='utf-8') as file:
            self.documents = json.load(file)

    def load_index(self):
        self.index = faiss.read_index(self.index_file)
        self.embeddings = self.model.encode([doc['text'] for doc in self.documents])
        self.tfidf_vectorizer.fit([doc['text'] for doc in self.documents])

    def create_index(self):
        self.embeddings = self.model.encode([doc['text'] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype('float16'))
        faiss.write_index(self.index, self.index_file)
        self.tfidf_vectorizer.fit([doc['text'] for doc in self.documents])

    def preprocess_financial_data(self, text):
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text

    def get_relevant_documents(self, query, top_k=20):
        query_embedding = self.model.encode([query])[0].astype('float16').reshape(1, -1)
        D, I = self.index.search(query_embedding, top_k)
        
        relevant_docs = [self.documents[i] for i in I[0]]
        
        # Re-rank using TF-IDF and recency
        tfidf_scores = self.tfidf_vectorizer.transform([doc['text'] for doc in relevant_docs])
        query_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_similarities = (tfidf_scores * query_tfidf.T).toarray().flatten()
        
        now = datetime.now(timezone.utc)
        recency_scores = []
        for doc in relevant_docs:
            doc_date = datetime.fromisoformat(doc['date'])
            if doc_date.tzinfo is None:
                doc_date = doc_date.replace(tzinfo=timezone.utc)
            days_diff = (now - doc_date).days
            recency_scores.append(1 / 1.2 * (days_diff + 1))  # Adding 1 to avoid division by zero
        
        combined_scores = 0.3 * tfidf_similarities + 0.7 * np.array(recency_scores)
        ranked_indices = combined_scores.argsort()[::-1]
        
        return [relevant_docs[i] for i in ranked_indices]
    
    def fetch_currency_data(self, base_currency, target_currencies):
        base_url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies"
        fallback_url = "https://latest.currency-api.pages.dev/v1/currencies"
        
        try:
            response = requests.get(f"{base_url}/{base_currency}.json")
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            response = requests.get(f"{fallback_url}/{base_currency}.json")
            response.raise_for_status()
            data = response.json()

        return {currency: data[base_currency][currency] for currency in target_currencies if currency in data[base_currency]}

    def update_documents(self, new_messages):
        # Add Russia Central Bank stack
        url = "https://www.cbr.ru/hd_base/keyrate/"
        response = requests.get(url)
        response.raise_for_status()
        
        df = pd.read_html(response.text)[0]
        df.columns = ['Date', 'Rate']
        latest_record = df.iloc[0]

        cb_date = datetime.strptime(latest_record['Date'], "%d.%m.%Y").replace(tzinfo=timezone.utc)
        cb_rate = float(latest_record['Rate'] / 100)
        
        cb_message = {
            "text": f"Текущая ставка центрального банка ЦБ России РФ составляет {cb_rate} процентов.",
            "link": "https://www.cbr.ru/hd_base/keyrate/",
            "date": cb_date.isoformat()
        }
        
        new_messages.insert(0, cb_message)

        # Fetch currency rates
        # Source for currencies https://github.com/fawazahmed0/exchange-api?tab=readme-ov-file#endpoints 
        target_currencies = ['rub', 'usd', 'eur', 'cny', 'jpy', 'gbp']
        base_currency = 'usd'
        currency_data = self.fetch_currency_data('usd', target_currencies)
        currency_message = {
            "text": f"Курсы валют на сегодняшний день к {base_currency.upper()} - {', '.join([f'{currency.upper()}: {rate}' for currency, rate in currency_data.items()])}.",
            "link": "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json",
            "date": datetime.now(timezone.utc).isoformat()
        }
        
        new_messages.insert(1, currency_message)

        # Fetch Bitcoin rate
        bitcoin_data = self.fetch_currency_data('btc', ['usd'])
        bitcoin_message = {
            "text": f"Курс биткоина (Bitcoin) в долларах на сегодняшний день: USD: {bitcoin_data.get('usd', 'N/A')}.",
            "link": "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/btc.json",
            "date": datetime.now(timezone.utc).isoformat()
        }
        
        new_messages.insert(2, bitcoin_message)

        spam_words = {'аудиоверсия', 'скидка', 'реклама', 'промокод'}
        for msg in new_messages:
            if any(word in msg['text'].lower() for word in spam_words):
                msg['text'] = ''
            date = datetime.fromisoformat(msg['date'])
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            msg['date'] = date.isoformat()
        
        self.documents = new_messages
        with open(self.data_file, 'w', encoding='utf-8') as file:
            json.dump(self.documents, file, ensure_ascii=False, indent=4)
        self.create_index()

