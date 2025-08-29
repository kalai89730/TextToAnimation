
#None selected

#Skip to content
#Using Gmail with screen readers

#Conversations
#14.2 GB of 15 GB (94%) used
##Terms · Privacy · Program Policies
#Last account activity: 5 hours ago
#Details
# import chainlit as cl
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_community.llms import Ollama
# import spacy
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.chains import RetrievalQA, LLMChain
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from langchain.prompts import ChatPromptTemplate

# from langchain_ollama import OllamaLLM

# # Set Chainlit Markdown Path
# os.environ["CHAINLIT_MARKDOWN_PATH"] = "chainlit.md"

# # Function to lemmatize text content
# def lemmatize_content(text):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
#     return " ".join([token.lemma_ for token in doc])

# # Load PDFs from the dataset

# # Load text file
# def load_text_file(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return f.read()

# # Directory containing text files
# text_folder = "extracted_text"
# documents = []

# # Load all text files
# for file in os.listdir(text_folder):
#     if file.endswith(".txt"):
#         file_path = os.path.join(text_folder, file)
#         content = load_text_file(file_path)
#         documents.append(Document(page_content=content, metadata={"source": file}))
# # loader = PyPDFDirectoryLoader("data3")
# # docs = loader.load()

# # Load spaCy model
# nlp = spacy.load("en_core_web_sm")

# # Lemmatize documents
# lemmatized_docs = [{"content": lemmatize_content(doc.page_content)} for doc in documents]
    
# # Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=25)
# from langchain.schema import Document

# # Convert lemmatized text into Document objects
# lemmatized_docs_as_documents = [
#     Document(page_content=doc["content"]) for doc in lemmatized_docs
# ]
# chunks = text_splitter.split_documents(lemmatized_docs_as_documents)

# # Load embeddings model
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
# embeddings = SentenceTransformerEmbeddings(model_name="jinaai/jina-embeddings-v2-base-code")

# # Create vector database
# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     collection_name='text-to-animation-chunks'
# )
# print("Vector database created successfully")

# # Initialize LLM with DeepSeek Model
# llm = Ollama(
#     model="hf.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF:Q4_K_S",
#     temperature=0.5,   # Controls creativity (Lower = More deterministic)
#     top_p=0.9,
#     num_ctx=4096
# )

# # MultiQueryRetriever for enhanced search accuracy
# Query_prompt = PromptTemplate(
#     input_variables=['question'],
#     template="""
#     You are an AI assistant. Generate 2 different versions of the given user question 
#     to retrieve relevant documents from a vector database related to text-to-animation.
    
#     Original Question: {question}
#     """
# )
# retriever = MultiQueryRetriever.from_llm(
#     vectorstore.as_retriever(),
#     llm,
#     prompt=Query_prompt
# )

# # Custom Persona Prompt for the Project
# prompt_template = """.

# You are currently working on a *Text-to-Animation System* that converts textual descriptions into animated 3D models using *Hugging Face LLMs and Blender*.
# you should only give python blender script for animation creation based on the given user input
# ### **Context*8 ###
# {context}

# ### *Question:*
# {question}

# ### *Your Answer in python blender script:*
# """

# prompt = ChatPromptTemplate.from_template(prompt_template)

# # Define RAG Pipeline
# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# @cl.on_chat_start
# async def start():
#     await cl.Message(content="Automated 3D Animation using textual description").send()

# @cl.on_message
# async def handle_message(message: cl.Message):
#     user_query = message.content
#     result = chain.invoke(user_query)
#     await cl.Message(content=result).send()

import chainlit as cl
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.llms import Ollama
import spacy
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document  # ✅ FIXED: Importing Document before using it
import json
from langchain_ollama import OllamaLLM

# Set Chainlit Markdown Path
os.environ["CHAINLIT_MARKDOWN_PATH"] = "chainlit.md"

# Function to lemmatize text content
def lemmatize_content(text):
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Load dataset from JSON file
json_file_path = "C:\\Users\\kalaivani\\Desktop\\Genai\\animation\\blender_scripts_dataset.json"
with open(json_file_path, "r", encoding="utf-8") as json_file:
    dataset = json.load(json_file)

# Convert JSON data into Document objects
documents = [Document(page_content=item["Blender Script"], metadata={"Text Input": item["Text Input"]}) for item in dataset]


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Lemmatize documents
lemmatized_docs = [{"content": lemmatize_content(doc.page_content)} for doc in documents]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)

# Convert lemmatized text into Document objects
lemmatized_docs_as_documents = [
    Document(page_content=doc["content"]) for doc in lemmatized_docs
]
chunks = text_splitter.split_documents(lemmatized_docs_as_documents)

# Load embeddings model
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
embeddings = SentenceTransformerEmbeddings(model_name="jinaai/jina-embeddings-v2-base-code")

# Create vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name='text-to-animation-chunks'
)
print("Vector database created successfully")

# Initialize LLM with DeepSeek Model
llm = Ollama(
    model="hf.co/TheBloke/deepseek-llm-7B-chat-GGUF:Q2_K",
    temperature=0.4,
    top_p=0.9,
    num_ctx=4096
)

# MultiQueryRetriever for enhanced search accuracy
Query_prompt = PromptTemplate(
    input_variables=['question'],
    template="""
    You are an AI assistant. Generate 2 different versions of the given user question 
    to retrieve relevant documents from a vector database related to text-to-animation.
    
    Original Question: {question}
    """
)
retriever = MultiQueryRetriever.from_llm(
    vectorstore.as_retriever(),
    llm,
    prompt=Query_prompt
)

# Custom Persona Prompt for the Project
prompt_template = """
You are currently working on a *Text-to-Animation System* that converts textual descriptions into animated 3D models using *Hugging Face LLMs and Blender*.
You should only give Python Blender script for animation creation based on the given user input.

### *Context:*
{context}

### *Question:*
{question}

"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Define RAG Pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@cl.on_chat_start
async def start():
    await cl.Message(content="Automated 3D Animation using textual description").send()

@cl.on_message
async def handle_message(message: cl.Message):
    user_query = message.content
    result = chain.invoke(user_query)
    await cl.Message(content=result).send()
if __name__ == "__main__":
    cl.run()      