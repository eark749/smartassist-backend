from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import json
import os
import uuid

# LangChain imports
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS clients
s3 = boto3.client('s3', region_name='eu-north-1')

# LangChain Bedrock LLM - Qwen2.5 32B
llm = ChatBedrock(
    model_id="qwen2-5-32b-instruct-v1:0",
    region_name="eu-north-1",
    model_kwargs={"temperature": 0.7, "max_tokens": 2048}
)

# Bedrock Embeddings - Use us-east-1 for Titan v2
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

# OpenSearch configuration
OPENSEARCH_ENDPOINT = 'vpc-smartassist-search1-u56375uz44djiy5akq47vbvikm.eu-north-1.es.amazonaws.com'
OPENSEARCH_REGION = 'eu-north-1'

# Global vector store
vector_store = None

def get_opensearch_auth():
    """Get AWS authentication for OpenSearch"""
    credentials = boto3.Session().get_credentials()
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        OPENSEARCH_REGION,
        'es',
        session_token=credentials.token
    )

def get_vector_store():
    """Initialize OpenSearch vector store with LangChain"""
    global vector_store
    if vector_store is None:
        try:
            http_auth = get_opensearch_auth()
            vector_store = OpenSearchVectorSearch(
                opensearch_url=f"https://{OPENSEARCH_ENDPOINT}",
                index_name="documents",
                embedding_function=embeddings,
                http_auth=http_auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
            return None
    return vector_store

def search_documents(query: str) -> str:
    """
    Tool to search uploaded documents using RAG.
    Use this when the user asks about specific documents, files, or uploaded content.
    """
    try:
        vs = get_vector_store()
        if vs is None:
            return "No documents have been uploaded yet. Please upload documents first."
        
        # Search for relevant documents
        docs = vs.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant documents found for your query."
        
        # Combine document content
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Relevant information from uploaded documents:\n\n{context}"
    
    except Exception as e:
        return f"Unable to search documents at this time. Error: {str(e)}"

# Define tools for the agent
tools = [
    Tool(
        name="SearchDocuments",
        func=search_documents,
        description="Useful for when you need to answer questions about uploaded documents, files, or specific content that users have provided. Input should be a search query."
    )
]

# Agent prompt template
agent_prompt = PromptTemplate.from_template("""You are a helpful AI assistant with access to uploaded documents.

You have access to the following tools:

{tools}

Tool Names: {tool_names}

When a user asks about uploaded documents, files, or specific content, use the SearchDocuments tool.
For general questions, answer directly using your knowledge.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# Create the agent
agent = create_react_agent(llm, tools, agent_prompt)

# Agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

class ChatRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to S3 (Lambda will process it)"""
    try:
        file_id = str(uuid.uuid4())
        bucket_name = 'smartassist-documents-560615486015'
        
        # Upload to S3
        s3.upload_fileobj(
            file.file,
            bucket_name,
            f"{file_id}/{file.filename}"
        )
        
        return {
            "message": "File uploaded successfully",
            "file_id": file_id,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with intelligent agent that decides when to use RAG.
    The agent will automatically search documents when relevant.
    """
    try:
        # Try agent first
        try:
            response = agent_executor.invoke({"input": request.query})
            output = response.get("output", "")
            if output:
                return {
                    "response": output,
                    "agent_used": True
                }
        except Exception as agent_error:
            print(f"Agent failed: {agent_error}, falling back to direct LLM")
        
        # Fallback to direct LLM
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=request.query)])
        
        return {
            "response": response.content,
            "agent_used": False
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

@app.get("/")
def root():
    return {
        "message": "SmartAssist Backend API",
        "version": "2.0 - LangChain Agent",
        "endpoints": {
            "/health": "Health check",
            "/chat": "Chat with AI agent (POST)",
            "/upload": "Upload documents (POST)"
        }
    }
