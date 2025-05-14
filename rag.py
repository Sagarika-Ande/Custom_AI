# rag_pdf_google_example.py

import os
from dotenv import load_dotenv

# --- Langchain components for Google Generative AI ---
from langchain_community.document_loaders import PyPDFLoader # MODIFIED: For PDF loading
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Main script execution ---
def main():
    # --- 1. Load Environment Variables ---
    load_dotenv()
    print("Loaded environment variables.")

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
        return

    # --- 2. Define Path to Your PDF Document ---
    pdf_file_path = "sodapdf-converted.pdf" # MODIFIED: Path to your PDF

    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file not found at '{pdf_file_path}'. Please place your PDF there or update the path.")
        return
    print(f"Attempting to load PDF from: {pdf_file_path}")


    # --- 3. Load Documents using PyPDFLoader ---
    try:
        # Using PyPDFLoader to load content from the PDF file
        # Each page of the PDF will typically become a separate Document object
        pdf_loader = PyPDFLoader(pdf_file_path)
        documents = pdf_loader.load() # This might take a moment for larger PDFs
    except Exception as e:
        print(f"Error loading PDF: {e}")
        print("Ensure the 'pypdf' library is installed ('pip install pypdf') and the PDF file is not corrupted.")
        return

    if not documents:
        print("No documents were loaded from the PDF. The PDF might be empty or unreadable.")
        return
    print(f"Loaded {len(documents)} pages/documents from the PDF.")
    # print(f"First page content snippet: {documents[0].page_content[:200]}...") # Optional: print snippet


    # --- 4. Split Documents into Manageable Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjusted chunk size, PDF text can be denser
        chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(documents)

    if not doc_splits:
        print("No document splits were created. This might happen if PDF content is too small or processing failed.")
        return
    print(f"Split PDF content into {len(doc_splits)} chunks.")

    # --- 5. Create Text Embeddings and Store in a Vector Store ---
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Initialized GoogleGenerativeAIEmbeddings.")

        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=embeddings_model
        )
        print("Vector store (FAISS) created successfully with PDF content.")
    except Exception as e:
        print(f"Error creating embeddings model or vector store: {e}")
        # (Error message from previous script is good)
        return

    # --- 6. Create a Retriever from the Vector Store ---
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks for PDF
    print("Retriever created.")

    # --- 7. Define the RAG Prompt Template ---
    RAG_PROMPT_TEMPLATE = """
CONTEXT:
{context}

QUESTION:
{question}

Based ONLY on the provided CONTEXT from the PDF document, answer the QUESTION.
If the context does not contain the answer, state that the information is not found in the provided PDF excerpts.
Be concise and directly answer the question using information from the context.
"""
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    print("RAG prompt template created.")

    # --- 8. Define the Language Model (LLM) ---
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2, convert_system_message_to_human=True)
        print(f"Initialized ChatGoogleGenerativeAI with model 'gemini-1.5-flash-latest'.")
    except Exception as e:
        print(f"Error initializing ChatGoogleGenerativeAI: {e}")
        return

    # --- 9. Create the RAG Chain using LangChain Expression Language (LCEL) ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain created.")

    # --- 10. Ask Questions using the RAG Chain (Tailor these to your PDF content) ---
    print("\n--- Ready to answer questions about your PDF! ---")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nAsk a question about the PDF: ")
        if question.lower() in ['exit', 'quit']:
            break
        if not question.strip():
            continue

        try:
            print("Processing your question...")
            answer = rag_chain.invoke(question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error invoking RAG chain for question '{question}': {e}")
            print("This could be due to API rate limits, network issues, or problems with the model response.")

    print("\nExiting PDF Q&A session.")

if __name__ == "__main__":
    main()