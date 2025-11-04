from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opensearchpy import OpenSearch
import boto3
import json
import os
import uuid

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bedrock = boto3.client('bedrock-runtime', region_name='eu-north-1')
s3 = boto3.client('s3', region_name='eu-north-1')
dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')

# OpenSearch client will be initialized on first use
opensearch_client = None

def get_opensearch_client():
    global opensearch_client
    if opensearch_client is None:
        opensearch_endpoint = 'vpc-smartassist-search1-u56375uz44djiy5akq47vbvikm.eu-north-1.es.amazonaws.com'
        
        opensearch_client = OpenSearch(
            hosts=[{'host': opensearch_endpoint, 'port': 443}],
            http_auth=(os.environ.get('OPENSEARCH_USER', 'admin'), os.environ.get('OPENSEARCH_PASSWORD', 'admin')),
            use_ssl=True,
            verify_certs=True,
            ssl_show_warn=False
        )
    return opensearch_client

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
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
        "file_id": file_id
    }

@app.post("/chat")
async def chat(query: str):
    # Generate query embedding
    embedding_response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        body=json.dumps({"inputText": query})
    )
    query_embedding = json.loads(embedding_response['body'].read())['embedding']
    
    # Get OpenSearch client
    os_client = get_opensearch_client()
    
    # Search similar documents in OpenSearch
    search_results = os_client.search(
        index='documents',
        body={
            'query': {
                'knn': {
                    'embeddings': {
                        'vector': query_embedding,
                        'k': 3
                    }
                }
            }
        }
    )
    
    # Get relevant context
    context = ""
    for hit in search_results['hits']['hits']:
        context += hit['_source']['text'] + "\n\n"
    
    # Generate response using Claude
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    response = bedrock.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    )
    
    result = json.loads(response['body'].read())
    answer = result['content'][0]['text']
    
    return {
        "response": answer,
        "sources": search_results['hits']['hits']
    }
