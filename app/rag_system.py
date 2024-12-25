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
import pickle

from datetime import datetime
# from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
import openai
openai.api_key = api_key

from openai import OpenAI

class RAGSystem:
    def __init__(self, data_file='data/telegram_messages.json', index_file='data/faiss_index.idx', cache_file='data/embedding_cache.pkl'):
        self.client = OpenAI()
        self.documents = []
        self.embeddings = None
        self.data_file = data_file
        self.index_file = index_file
        self.cache_file = cache_file
        self.embedding_cache = self.load_embedding_cache()
        self.index = None
        self.load_or_create_index()

    def load_embedding_cache(self):
        """Load the embedding cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_embedding_cache(self):
        """Save the embedding cache to disk."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.embedding_cache, f)

    def get_embedding(self, text):
        cached_embeddings = []
        texts_to_embed = []
        
        for t in text:
            if t in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[t])
            else:
                texts_to_embed.append(t)
        
        if not texts_to_embed:
            return cached_embeddings
            
        try:
            batch_size = 100
            text_batches = [texts_to_embed[i:i + batch_size] for i in range(0, len(texts_to_embed), batch_size)]
            new_embeddings = []
            
            for batch in text_batches:
                response = OpenAI().embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                
                for t, emb in zip(batch, batch_embeddings):
                    self.embedding_cache[t] = emb
                    new_embeddings.append(emb)
            
            self.save_embedding_cache()
            
            final_embeddings = []
            new_emb_idx = 0
            
            for t in text:
                if t in self.embedding_cache:
                    final_embeddings.append(self.embedding_cache[t])
                else:
                    final_embeddings.append(new_embeddings[new_emb_idx])
                    new_emb_idx += 1
                    
            return final_embeddings
            
        except Exception as e:
            print(f"Error getting embedding: {e} {batch}")
            return None

    def load_or_create_index(self):
        if os.path.exists(self.data_file):
            self.load_documents()
            
            if os.path.exists(self.index_file):
                try:
                    self.index = faiss.read_index(self.index_file)
                    if self.index.ntotal == len([doc for doc in self.documents if doc['text'] != '']):
                        print("Existing index loaded successfully")
                        return
                except Exception as e:
                    print(f"Error loading index: {e}")
            
            self.create_index()
        else:
            print("Data file not found. Please update the database.")

    def load_documents(self):
        with open(self.data_file, 'r', encoding='utf-8') as file:
            self.documents = json.load(file)

    def load_index(self):
        try:
            self.index = faiss.read_index(self.index_file)
            all_texts = [doc['text'] for doc in self.documents if doc['text'] != '']
            self.embeddings = np.array(self.get_embedding(all_texts))
        except Exception as e:
            print(f"Error loading index: {e}")
            self.create_index()

    def create_index(self):
        try:
            all_texts = [doc['text'] for doc in self.documents if doc['text'] != '']
            self.embeddings = np.array(self.get_embedding(all_texts))
            dimension = len(self.embeddings[0]) if len(self.embeddings) > 0 else 1536
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))
            faiss.write_index(self.index, self.index_file)
        except Exception as e:
            print(f"Error creating index: {e}")

    def preprocess_financial_data(self, text):
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text

    def get_relevant_documents(self, query, top_k=20):
        query_embedding = np.array(self.get_embedding([query])[0]).astype('float32').reshape(1, -1)
        D, I = self.index.search(query_embedding, top_k)
        
        relevant_docs = [self.documents[int(i)] for i in I[0]]
        
        now = datetime.now(timezone.utc)
        recency_scores = []
        for doc in relevant_docs:
            doc_date = datetime.fromisoformat(doc['date'])
            if doc_date.tzinfo is None:
                doc_date = doc_date.replace(tzinfo=timezone.utc)
            days_diff = (now - doc_date).days
            recency_scores.append(1 / (1.2 * (days_diff + 1)))  # Adding 1 to avoid division by zero
        
        combined_scores = 0.3 * D[0] + 0.7 * np.array(recency_scores)
        ranked_indices = np.argsort(combined_scores)[::-1]
        
        return [relevant_docs[int(i)] for i in ranked_indices]
    
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
            "text": f"Курсы валют на сегодняшний день {datetime.now().isoformat()} к {base_currency.upper()} - {', '.join([f'{currency.upper()}: {rate}' for currency, rate in currency_data.items()])}.",
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

        old_docs = self.documents.copy() if self.documents else []
        self.documents = new_messages
        
        with open(self.data_file, 'w', encoding='utf-8') as file:
            json.dump(self.documents, file, ensure_ascii=False, indent=4)
        
        if len(old_docs) != len(self.documents) or any(old != new for old, new in zip(old_docs, self.documents)):
            self.create_index()
