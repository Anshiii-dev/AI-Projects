from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import os
from pypdf import PdfReader
from dotenv import load_dotenv
import re
from typing import List, Dict, Any
import easyocr
load_dotenv("my.env")
os.environ["GROQ_API_KEY"] = os.getenv("Groq")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "wus2jsnwjs"

llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.3)
search_llm = ChatGroq(model="compound-beta", temperature=0.3)

# ==================== PDF PROCESSING FUNCTIONS ====================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        pdf = PdfReader(pdf_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean and preprocess extracted text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text.strip()

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for better processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

def create_embeddings_and_vectorstore(text_chunks: List[str]) -> FAISS:
    """Create embeddings and store in FAISS vector database."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def search_document(vectorstore: FAISS, query: str, k: int = 3) -> List[str]:
    """Search for relevant document chunks based on query."""
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

# ==================== PROMPT TEMPLATES ====================

# Main PDF analysis prompt
pdf_prompt = PromptTemplate(
    template="""You are an expert document analyst and AI assistant. Your task is to analyze PDF documents and provide comprehensive, accurate answers based on the document content.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided document context
2. If the information is not in the document, clearly state that
3. Provide specific references to the document content when possible
4. Be concise but thorough
5. If the question requires information beyond the document, suggest what additional sources might be needed

ANSWER:""",
    input_variables=["context", "query"]
)

# Document summary prompt
summary_prompt = PromptTemplate(
    template="""You are an expert document summarizer. Create a comprehensive summary of the following document content.

DOCUMENT CONTENT:
{text}

INSTRUCTIONS:
1. Create a structured summary with key points
2. Identify main topics and themes
3. Highlight important facts, figures, or conclusions
4. Keep the summary concise but informative
5. Organize information logically

SUMMARY:""",
    input_variables=["text"]
)

# Document Q&A prompt
qa_prompt = PromptTemplate(
    template="""You are a helpful assistant answering questions about a specific document. Use only the information provided in the document context to answer questions.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided document context
- If the answer is not in the document, say "I cannot find this information in the document"
- Be specific and reference relevant parts of the document
- Provide clear, concise answers

ANSWER:""",
    input_variables=["context", "question"]
)

# Decision prompt for routing
decision_prompt = PromptTemplate(
    template="""You are a decision assistant. Determine what type of processing is needed for the user's request.

USER QUERY: {query}

OPTIONS:
1. "pdf_analysis" - If the user wants to analyze or ask questions about a PDF document
2. "internet_search" - If the user needs current information not in documents
3. "general_qa" - If the user has a general question not requiring documents

Respond with only one of: "pdf_analysis", "internet_search", or "general_qa"

DECISION:""",
    input_variables=["query"]
)

# ==================== CHAIN SETUP ====================

# Create chains
decision_chain = decision_prompt | llm | StrOutputParser()
pdf_chain = pdf_prompt | llm | StrOutputParser()
summary_chain = summary_prompt | llm | StrOutputParser()
qa_chain = qa_prompt | llm | StrOutputParser()
general_chain = PromptTemplate(
    template="You are a helpful AI assistant. Answer the following question: {query}",
    input_variables=["query"]
) | llm | StrOutputParser()
search_chain = PromptTemplate(
    template="You are a search assistant. Answer based on search results: {query}",
    input_variables=["query"]
) | search_llm | StrOutputParser()

# ==================== MAIN PROCESSING FUNCTION ====================
def process_images(path,query):
    text=easyocr.Reader(path)
    cleaned_text=clean_text(text)
    print("Generating answer...")
    result = pdf_chain.invoke({"context": cleaned_text, "query": query})
        
    return result


def process_pdf_query(pdf_path: str, query: str) -> str:
    """Main function to process PDF queries."""
    try:
        # Extract and process PDF
        print("Extracting text from PDF...")
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            return "Error: Could not extract text from PDF"
        
        # Clean text
        cleaned_text = clean_text(raw_text)
        
        # Split into chunks
        print("Processing document chunks...")
        text_chunks = split_text_into_chunks(cleaned_text)
        
        # Create vector store
        print("Creating search index...")
        vectorstore = create_embeddings_and_vectorstore(text_chunks)
        
        # Search for relevant context
        relevant_chunks = search_document(vectorstore, query)
        context = "\n\n".join(relevant_chunks)
        
        # Generate answer
        print("Generating answer...")
        result = pdf_chain.invoke({"context": context, "query": query})
        
        return result
        
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def get_document_summary(pdf_path: str) -> str:
    """Get a summary of the PDF document."""
    try:
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            return "Error: Could not extract text from PDF"
        
        cleaned_text = clean_text(raw_text)
        # Use first 2000 characters for summary to avoid token limits
        summary_text = cleaned_text[:2000] + "..." if len(cleaned_text) > 2000 else cleaned_text
        
        result = summary_chain.invoke({"text": summary_text})
        return result
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# ==================== MAIN INTERFACE ====================

def main():
    """Main interface for the chatbot."""
    print("=== AI Chatbot with PDF Analysis ===")
    print("1. Ask general questions")
    print("2. Analyze PDF document")
    print("3. Get PDF summary")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        query = input("Enter your question: ")
        decision = decision_chain.invoke({"query": query})
        print(f"Decision: {decision}")
        
        if "pdf_analysis" in decision.lower():
            pdf_path = input("Enter PDF file path: ")
            result = process_pdf_query(pdf_path, query)
        elif "internet_search" in decision.lower():
            result = search_chain.invoke({"query": query})
        else:
            result = general_chain.invoke({"query": query})
        
        print(f"Answer: {result}")
        
    elif choice == "2":
        pdf_path = input("Enter PDF file path: ")
        query = input("Enter your question about the PDF: ")
        result = process_pdf_query(pdf_path, query)
        print(f"Answer: {result}")
        
    elif choice == "3":
        pdf_path = input("Enter PDF file path: ")
        summary = get_document_summary(pdf_path)
        print(f"Summary: {summary}")
        
    elif choice == "4":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
      