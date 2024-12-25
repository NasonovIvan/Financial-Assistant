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
        # self.tfidf_vectorizer = TfidfVectorizer()
        self.embedding_cache = self.load_embedding_cache()
        self.load_or_create_index()

    def load_embedding_cache(self):
        # print('load_embedding_cache')
        """Load the embedding cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_embedding_cache(self):
        """Save the embedding cache to disk."""
        # print('save_embedding_cache')
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.embedding_cache, f)

    @classmethod
    def get_embedding(self, text):
        # for t in text:
        #     if t in self.embedding_cache:
        #         return self.embedding_cache[t]
        # print('get_embedding')
        try:
            batch_size = 100
            text_batches = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]
            embeddings = []
            for batch in text_batches:
                response = OpenAI().embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                embeddings += [data.embedding for data in response.data]

            # self.embedding_cache[text] = embedding
            # self.save_embedding_cache()  # Save the updated cache to disk
            return embeddings
        
        
        except Exception as e:
            print(f"Error getting embedding: {e} {batch}")
            return None

    def load_or_create_index(self):
        if os.path.exists(self.data_file) and os.path.exists(self.index_file):
            self.load_documents()
            # self.load_index()
            self.create_index()
        else:
            print("Index or data file not found. Please update the database.")

    def load_documents(self):
        with open(self.data_file, 'r', encoding='utf-8') as file:
            self.documents = json.load(file)

    # def load_index(self):
    #     self.index = faiss.read_index(self.index_file)
    #     self.embeddings = self.model.encode([doc['text'] for doc in self.documents])
    #     self.tfidf_vectorizer.fit([doc['text'] for doc in self.documents])

    def load_index(self):
        # print('load_index')
        try:
            self.index = faiss.read_index(self.index_file)
            all_texts = [doc['text'] for doc in self.documents if doc['text'] != '']
            self.embeddings = np.array(self.get_embedding(all_texts))
        except Exception as e:
            print(f"Error loading index: {e}")
            self.create_index()

    # def create_index(self):
    #     self.embeddings = self.model.encode([doc['text'] for doc in self.documents])
    #     self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
    #     self.index.add(self.embeddings.astype('float16'))
    #     faiss.write_index(self.index, self.index_file)
    #     self.tfidf_vectorizer.fit([doc['text'] for doc in self.documents])

    def create_index(self):
        # print('create_index')
        try:
            all_texts = [doc['text'] for doc in self.documents if doc['text'] != '']
            self.embeddings = np.array(self.get_embedding(all_texts))
            dimension = len(self.embeddings[0]) if len(self.embeddings) > 0 else 1536  # OpenAI embedding dimension
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings.astype('float32'))
            faiss.write_index(self.index, self.index_file)
            # self.index.search(np.expand_dims(self.embeddings[0], axis=0), 3)
        except Exception as e:
            print(f"Error creating index: {e}")

    def preprocess_financial_data(self, text):
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text

    def get_relevant_documents(self, query, top_k=20):
        query_embedding = np.array(self.get_embedding(query)).astype('float32').reshape(1, -1)
        D, I = self.index.search(query_embedding, top_k)
        
        relevant_docs = [self.documents[i] for i in I[0]]

        # Re-rank using TF-IDF and recency
        # tfidf_scores = self.tfidf_vectorizer.transform([doc['text'] for doc in relevant_docs])
        # query_tfidf = self.tfidf_vectorizer.transform([query])
        # tfidf_similarities = (tfidf_scores * query_tfidf.T).toarray().flatten()
        
        now = datetime.now(timezone.utc)
        recency_scores = []
        for doc in relevant_docs:
            doc_date = datetime.fromisoformat(doc['date'])
            if doc_date.tzinfo is None:
                doc_date = doc_date.replace(tzinfo=timezone.utc)
            days_diff = (now - doc_date).days
            recency_scores.append(1 / 1.2 * (days_diff + 1))  # Adding 1 to avoid division by zero
        
        combined_scores = 0.3 * np.array(D) + 0.7 * np.array(recency_scores)
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
        
        self.documents = new_messages
        with open(self.data_file, 'w', encoding='utf-8') as file:
            json.dump(self.documents, file, ensure_ascii=False, indent=4)
        self.create_index()

rag_system = RAGSystem()

rag_system.get_relevant_documents('Ставка Цб РФ?')
# batch_test = ['**Отменили комиссии за переводы и платежи для малого бизнеса по всей стране** 🤑\n\n',
# '[Откройте счёт в Альфа-Банке](https://alfabank.ru/sme/agent/free-transfers/) доконца года — и заберите бесплатную подписку на переводы. \n\n',
# 'Подписку подключим автоматически — на **3 месяца**.А вместе с ней **удвоим лимиты на переводы** юрлицам, физлицам и себе со счёта ИП 🗓\n\n',
# 'Сэкономите на комиссии и сможетесосредоточиться на важных задачах бизнеса 📖\n \n@aaaa_business']
# RAGSystem.get_embedding(batch_test)