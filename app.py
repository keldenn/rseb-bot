from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import os
import pickle
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from brain import (
    generate_response,
    update_conversation_history,
    generate_streaming_response,
    get_index_for_pdf,
    load_faiss_index,
    load_pdfs_from_backend,
    get_pdf_hash,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PROMPT_TEMPLATE = """
You are a helpful Assistant who answers user questions based on multiple contexts given to you. 

- Respond **Kuzu Zangpola! How can I assist you today?** to greetings like hello, hi, hey, wai, kuzu, kuzu zangpo, or kuzu zangpola.
- Keep your answer short and to the point.
- The provided evidence is extracted from PDF content with metadata.
- Focus on metadata when answering.
- If the document **does not contain relevant information**, reply **Sorry la, the question is beyond my domain knowledge.**.
  "You are an AI customer service Assistant created by Royal Security Exchange of Bhutan to provide helpful and informative responses based on the information provided to me."
- **If the response involves a process, steps, or principles, format them as a properly spaced numbered list, ensuring each item appears on a new line with one blank line between them.**
- **If the response involves steps, principles, or numbered items, always follow this format:**

  1. First point.

  2. Second point.

  3. Third point.

  - Each numbered point must be on a separate line.
  - There must be a blank line between numbered items.
  - Do not combine multiple points into a single paragraph.

- If a response does not require a numbered list, reply in a normal paragraph format. 
- Always make sure you respond "I'm here to help, so if there's anything else you'd like to ask or discuss, please let me know!" when offensive
languages like jadha, jaddha, j, long, laro, po are used.

"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    message: str
    stream: bool = False

conversation_histories = {}

# ===================== FAISS Index Management =====================

def load_or_create_vectordb():
    """Loads FAISS index if exists, otherwise rebuilds it if PDFs change."""
    
    if not os.path.exists("index"):
        os.makedirs("index")  # Ensure folder exists

    # Load PDFs and calculate their hashes
    pdf_files, pdf_filenames = load_pdfs_from_backend()
    pdf_hashes = {filename: get_pdf_hash(filepath) for filename, filepath in zip(pdf_filenames, pdf_files)}

    # Define index and metadata paths
    index_path = "index/index.faiss"
    metadata_path = "index/metadata.pkl"

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        # Load saved metadata
        with open(metadata_path, "rb") as f:
            saved_metadata = pickle.load(f)

        # If no changes, load existing FAISS index
        if saved_metadata == pdf_hashes:
            return load_faiss_index(OPENAI_API_KEY)

    # If PDFs changed, rebuild the FAISS index
    print("⚠️ PDF changes detected! Updating FAISS index...")
    
    vectordb = get_index_for_pdf(pdf_files, pdf_filenames, OPENAI_API_KEY)

    # Save updated metadata
    with open(metadata_path, "wb") as f:
        pickle.dump(pdf_hashes, f)

    return vectordb

# Load FAISS index at startup
vectordb = load_or_create_vectordb()

# ===================== Chatbot API =====================
@app.get("/")
async def health_check():
    return "The health check success!"


@app.post("/chat/")
async def chat(request: QueryRequest, background_tasks: BackgroundTasks):
    """Handles chat requests with RAG-based retrieval."""
    
    if request.stream:
        return StreamingResponse(
            generate_streaming_response(
                request.message,
                OPENAI_API_KEY,
                PROMPT_TEMPLATE,
                conversation_histories.get(request.message, [])
            ),
            media_type="text/event-stream"
        )

    # Search FAISS index for relevant documents
    search_results = vectordb.similarity_search(request.message, k=3)

    # If relevant PDFs found, use them for answering
    if search_results:
        pdf_extract = "\n".join([result.page_content for result in search_results])
        system_prompt = PROMPT_TEMPLATE + f"\nPDF Content:\n{pdf_extract}"
    else:
        system_prompt = PROMPT_TEMPLATE

    response = generate_response(
        request.message, 
        OPENAI_API_KEY, 
        system_prompt,
        conversation_histories.get(request.message, [])
    )

    background_tasks.add_task(
        update_conversation_history,
        conversation_histories.get(request.message, []),
        request.message,
        response
    )

    return {"response": response}
 