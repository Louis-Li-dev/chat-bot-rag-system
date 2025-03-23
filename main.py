import os
import json
import markdown
import sqlite3
import datetime
from flask import Flask, render_template, request, redirect, url_for, abort
import google.generativeai as genai
from dotenv import load_dotenv
from model.RAG import RAGSystem

load_dotenv()
app = Flask(__name__)
api_key = os.environ.get("api_key")
system = RAGSystem()  # Loads documents and builds FAISS index

# Configure Gemini
genai.configure(api_key=api_key)
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 5000,
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# --- Database setup ---
DATABASE = os.path.join(os.getcwd(), 'data', 'chat_records.db')

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            chat_history TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_chat_record(name, chat_history):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    c.execute('INSERT INTO chat_records (name, chat_history, timestamp) VALUES (?, ?, ?)', 
              (name, json.dumps(chat_history), timestamp))
    conn.commit()
    conn.close()

def get_all_records():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT id, name, timestamp FROM chat_records ORDER BY id DESC')
    records = c.fetchall()
    conn.close()
    return records

def get_record(record_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT chat_history FROM chat_records WHERE id = ?', (record_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def delete_record(record_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('DELETE FROM chat_records WHERE id = ?', (record_id,))
    conn.commit()
    conn.close()

# Initialize the DB at startup
init_db()

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def chat():
    chat_history = []
    # Prepare document links from RAGSystem documents
    doc_links = []
    for idx, _ in enumerate(system.documents):
         doc_links.append({'id': idx, 'name': f"Document {idx+1}"})
    
    # If continuing a saved record, load it.
    record_param = request.args.get('record')
    if record_param:
        loaded_history = get_record(int(record_param))
        if loaded_history:
            chat_history = loaded_history

    chat_session = model.start_chat(history=[])
    
    if request.method == 'POST':
        # Save chat record branch.
        if 'save_record' in request.form:
            record_name = request.form['record_name']
            previous_history = request.form.get('chat_history', '[]')
            try:
                chat_history = json.loads(previous_history)
            except json.JSONDecodeError:
                chat_history = []
            # Trim assistant messages so they don't include the full document context.
            def trim_message(message):
                content = message.get('content', '')
                # If the assistant's message includes a long prompt with context,
                # split and keep only the answer portion.
                if message.get('role') == 'assistant' and "回答：" in content:
                    parts = content.split("回答：", 1)
                    content = "回答：" + parts[1].strip()
                # Optionally truncate if content is too long.
                limit = 500
                if len(content) > limit:
                    content = content[:limit] + "..."
                message['content'] = content
                return message
            chat_history = [trim_message(m.copy()) for m in chat_history]
            if record_name.strip():
                save_chat_record(record_name.strip(), chat_history)
            return redirect(url_for('chat'))
        else:
            user_input = request.form['user_input']
            selected_language = request.form['languageSelect']
            previous_history = request.form.get('chat_history', '[]')
            try:
                chat_history = json.loads(previous_history)
            except json.JSONDecodeError:
                chat_history = []
            chat_history.append({'role': 'user', 'content': user_input})
            
            # Generate a prompt that includes relevant documents automatically.
            response = system.generate_prompt(f"{user_input} (please reply in {selected_language})")
            response_text = get_ai_response(response, chat_session)
            chat_history.append({'role': 'assistant', 'content': response_text})
        
    return render_template(
        'chat.html',
        chat_history=chat_history,
        chat_history_json=chat_history,  # We'll use tojson filter in the template.
        databases=doc_links,
        records=get_all_records()
    )

def get_ai_response(message, chat_session):
    response = chat_session.send_message(message)
    return markdown.markdown(response.text)

@app.route('/document/<int:doc_id>')
def show_document(doc_id):
    if doc_id < 0 or doc_id >= len(system.documents):
        abort(404)
    document_content = system.documents[doc_id]
    return render_template('document.html', doc_id=doc_id, content=document_content)

@app.route('/record/<int:record_id>/delete')
def delete_chat_record(record_id):
    delete_record(record_id)
    return redirect(url_for('chat'))

@app.route('/record/<int:record_id>/continue')
def continue_chat(record_id):
    # Redirect to the main chat route with a query parameter that loads the record.
    return redirect(url_for('chat', record=record_id))

def main():
    app.run(port=int(os.environ.get('PORT', 0)))

if __name__ == "__main__":
    main()
