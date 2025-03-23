
# Chat Bot RAG System

A retrieval-augmented generation (RAG) chat bot system that combines the power of large language models with a document retrieval engine to provide context-aware and accurate conversational responses.

## Expected Results
<div align="center">

| Chat | Documents | Chat Records |
|------|-----------|--------------|
| <img src="https://github.com/user-attachments/assets/228d67eb-7763-4285-b619-a57877ca2be6" alt="Chat" width="200" /> | <img src="https://github.com/user-attachments/assets/b49727e3-170d-470b-b9d1-6b5fdfca4b03" alt="Documents" width="200" /> | <img src="https://github.com/user-attachments/assets/8598311d-338b-41b2-9b68-5c975176bada" alt="Chat Records" width="200" /> |



</div>


## Installation

Follow these steps to set up the Chat Bot RAG System locally:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Louis-Li-dev/chat-bot-rag-system.git
   cd chat-bot-rag-system
   ```

2. **Install dependencies:**

   The project uses Python; install the required packages with:

   ```bash
   pip install -r requirements.txt
   ```

   > _Note: Ensure you have Python 3.7 or higher installed._

3. **Create `.env`**
   - Create a `.env` file according to `.example.env`
   - Head to [Google AI Studio](https://aistudio.google.com/prompts/new_chat) Add your *Gemini* api key to your `.env`

4. **Put PDF files in `data/`**
   - Recommend you to put the files you want the model to compare and operate on under `data/`
## Usage

To run the chat bot system, execute the main script:

```bash
python main.py
```

## References

- **Gemini:**  
  *Gemini* is used as a core component in this system. (For more details, please refer to the [Gemini documentation](https://example.com/gemini) or the corresponding publication.)  

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2:**  
  This model is provided by the SentenceTransformers library. For additional details and usage, see the model card on [Hugging Face](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). For background on Sentence-BERT, refer to:  
  Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084).

- **Faiss:**  
  Faiss is a library for efficient similarity search and clustering of dense vectors. For more details, see:  
  Johnson, M., Douze, M., & Jégou, H. (2017). *Billion-scale similarity search with GPUs.* [arXiv:1702.08734](https://arxiv.org/abs/1702.08734) or visit the [Faiss GitHub repository](https://github.com/facebookresearch/faiss).
