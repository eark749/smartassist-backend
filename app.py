from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import json
import os

app = FastAPI()

bedrock = boto3.client('bedrock-runtime', region_name='eu-north-1')
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb', region_name='eu-north-1')

@app.get("/health")
def health_check():
    return {"status": "healthy"}

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
