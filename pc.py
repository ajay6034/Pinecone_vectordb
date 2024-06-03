from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import openai
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

loader=PyPDFDirectoryLoader("PDFs")

data=loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=150,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

test_chunks = text_splitter.split_documents(data)

len(test_chunks)


load_dotenv()

embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

index_name = "docs-quickstart-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
docsearch=vectorstore.add_documents(test_chunks)

retriever = vectorstore.as_retriever(search_kwargs={'k':1})

query = "yolo outperforms which model?"
docs=vectorstore.similarity_search(query)

llm = ChatOpenAI()

qa = RetrievalQA.from_chain_type(llm,
                                          chain_type='stuff',
                                          retriever=vectorstore.as_retriever(),
                                         
                                          )

query = "yolo outperforms which model?"

qa.run(query)