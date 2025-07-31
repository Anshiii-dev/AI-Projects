from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import load_tools, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
import os
import pyttsx3
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser


nature_prompt="""your task is to analyze the prompt and provide the nature of prompt like the user is talking about 
summary of the document or he is aksing about the topic from the document if he asking to summarize you should return a list which only contains the nature
of prompt for example [summary] or [topic] you should strictly follow this format and donot include any extra text only give the
reponse like this [summary] or [topic]"""


llm = ChatGoogleGenerativeAI(model="compound-beta", temperature=0.3)
def initialize_engine():
    import pyttsx3
    return pyttsx3.init()

def speak(engine, text):
    engine.say(text)
    engine.runAndWait()



    return os.getenv("GEMINI_API_KEY")

def create_chain(prompt):
    
    

    

    system_prompt = '''You are an AI chatbot. Your task is to respond to the user smartly. Provide the best and most detailed answers.'''

    prompt = PromptTemplate.from_messages([
        {"role": "AI Chatbot", "content": system_prompt},
        {"role": "User", "content": "{question}"}
    ])

    output_parser = StrOutputParser()
    chain=prompt | llm | output_parser, system_prompt
    response = chain.invoke({"question": prompt})
    return response
def document_reader(path,prompt):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    choice = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    nature=choice.invoke
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    db= FAISS.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    str_output_parser = StrOutputParser()
    QA_chain = retriever | llm | str_output_parser
    QA_chain.invoke({"question": "What is the main topic of the document?"})
    
      

def main():
    engine = initialize_engine()
    chain, system_prompt = create_chain()

    query = input("💬 Ask me anything: ")

    if query:
        response = chain.invoke({"question": system_prompt + query})
        print("Response:", response)
        speak_option = input("Do you want the response to be spoken? (yes/no): ")
        if speak_option.lower() == "yes":
            speak(engine, response)

if __name__ == "__main__":
    main()
