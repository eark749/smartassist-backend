from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import boto3
import json
import os
import uuid

app = FastAPI()

bedrock = boto3.client('bedrock-runtime', region_name='eu-north-1')
s3 = boto3.client('s3', region_name='eu-north-1')
dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')

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
def chat(query: str):
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps({
            "prompt": f"\n\nHuman: {query}\n\nAssistant:",
            "max_tokens_to_sample": 500
        })
    )
    return {"response": response}
