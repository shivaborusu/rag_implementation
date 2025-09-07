from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()


def get_query_response(query: str) -> str:
    model_name = "BAAI/bge-large-en-v1.5"  
    model_kwargs = {'device': 'cpu'}  
    encode_kwargs = {'normalize_embeddings': True}  
    embedder = HuggingFaceEmbeddings(  
    model_name=model_name,  
    model_kwargs=model_kwargs,  
    encode_kwargs=encode_kwargs
    )

    qdrant_client = QdrantClient(url=os.environ["QDRANT_URI"])

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name = "pdf_documents",
        embedding=embedder
        )
    
    retrieved_documents = vector_store.similarity_search(query=query)

    context = ""
    for doc in retrieved_documents:
        context += doc.page_content + "\n\n"

    llm = ChatGroq(model="llama-3.3-70b-versatile",
                   temperature=0,
                   max_tokens=100)
    
    prompt_template = """You are an useful assistant
    Answer {user_question} based on the {context}
    If no information is found, don't assume anything and say I don't know
    """

    prompt = PromptTemplate.from_template(prompt_template)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke(
        {"user_question":query,
         'context':context}
    )

    return response