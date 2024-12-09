import streamlit as st
from rag_system import RAGSystem
from gpt_client import get_chat_completion
from telegram_client import update_messages
import traceback
from datetime import datetime, timezone

rag_system = RAGSystem(data_file='data/telegram_messages.json', index_file='data/faiss_index.idx')

st.title("Financial Assistant")

def update_database():
    """Update the database with new messages."""
    try:
        new_messages = update_messages()
        rag_system.update_documents(new_messages)
        st.success("Database updated successfully!")
    except Exception as e:
        st.error(f"An error occurred while updating the database: {str(e)}")
        st.error(traceback.format_exc())

def generate_todays_summary():
    """Generate a summary of today's news."""
    try:
        today = datetime.now(timezone.utc).date()
        today_str = today.isoformat()
        todays_docs = [doc for doc in rag_system.documents if doc['date'].startswith(today_str)]

        if todays_docs:
            context = "\n".join([f"{doc['text']} [Link to Info]({doc['link']})" for doc in todays_docs])
            prompt = f"Summarize the following news in 3-4 sentences with reference links in text:\n\n{context}\n\nSummary:"
            summary = get_chat_completion(prompt)
            st.write(summary)

            st.divider()
            st.subheader("Sources:")
            for doc in todays_docs:
                st.markdown(f"• {doc['text']} [Link to Info]({doc['link']})")
        else:
            st.write("No news for today. Update Database")
    except Exception as e:
        st.error(f"An error occurred while generating the summary: {str(e)}")
        st.error(traceback.format_exc())

def answer_question(question):
    """Answer a user question about financial markets."""
    try:
        relevant_docs = rag_system.get_relevant_documents(question)
        context = "\n".join([f"{doc['text']} [Link to Info]({doc['link']})" for doc in relevant_docs])
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        answer = get_chat_completion(prompt)
        st.write(answer)

        st.divider()
        st.subheader("Sources:")
        for doc in relevant_docs:
            st.markdown(f"• {doc['text']} [Link to Info]({doc['link']})")
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")
        st.error(traceback.format_exc())

if st.button("Update Database"):
    with st.spinner("Updating database..."):
        update_database()

question = st.text_input("Ask a question about financial markets:")

if st.button("Get Today's Summary"):
    with st.spinner("Generating summary..."):
        generate_todays_summary()

if question:
    with st.spinner("Thinking..."):
        answer_question(question)

# Display images at the bottom
st.image("images/Picture1.png", use_column_width=True)
st.image("images/Picture2.png", use_column_width=True)
