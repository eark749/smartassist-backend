import json
import boto3
import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# Initialize AWS clients outside the handler for better performance
s3 = boto3.client('s3')
textract = boto3.client('textract')
bedrock_runtime = boto3.client('bedrock-runtime')

def lambda_handler(event, context):
    """
    AWS Lambda handler function to process an S3 event,
    extract text with Textract, generate embeddings with Bedrock,
    and index the data into OpenSearch.
    """
    try:
        # --- 1. Get S3 object details ---
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        print(f"Processing document: s3://{bucket}/{key}")
        
        # --- 2. Extract text using Textract ---
        textract_response = textract.detect_document_text(
            Document={'S3Object': {'Bucket': bucket, 'Name': key}}
        )
        
        # Combine all text
        document_text = ""
        for item in textract_response['Blocks']:
            if item['BlockType'] == 'LINE':
                document_text += item['Text'] + "\n"
        
        if not document_text:
            print("No text extracted by Textract. Aborting.")
            return {
                'statusCode': 400,
                'body': json.dumps('No text extracted from document.')
            }
        
        print(f"Extracted {len(document_text)} characters of text")
        
        # --- 3. Generate embeddings using Bedrock ---
        model_id = 'amazon.titan-embed-text-v1'
        
        # Titan has a max input length, chunk if needed
        max_length = 8000
        if len(document_text) > max_length:
            document_text = document_text[:max_length]
            print(f"Text truncated to {max_length} characters")
        
        embedding_payload = json.dumps({"inputText": document_text})
        embedding_response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=embedding_payload
        )
        
        response_body = embedding_response['body'].read()
        embeddings = json.loads(response_body.decode('utf-8'))['embedding']
        
        print(f"Generated embedding with {len(embeddings)} dimensions")
        
        # --- 4. Store in OpenSearch ---
        opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
        aws_region = os.environ.get('AWS_REGION', 'eu-north-1')
        
        if not opensearch_endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT environment variable is not set.")
        
        # Create SigV4 authentication
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            aws_region,
            'es',
            session_token=credentials.token
        )
        
        # Initialize OpenSearch client
        os_client = OpenSearch(
            hosts=[{'host': opensearch_endpoint, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        
        # Document to index
        document = {
            'document_id': key,
            'text': document_text,
            'embeddings': embeddings,
            'bucket': bucket,
            'filename': key.split('/')[-1]
        }
        
        # Index the document
        index_name = 'documents'
        doc_id = key.replace('/', '-')
        
        index_response = os_client.index(
            index=index_name,
            body=document,
            id=doc_id,
            refresh=True
        )
        
        print(f"Document indexed successfully: {index_response}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Document processed and indexed successfully',
                'document_id': doc_id,
                'text_length': len(document_text)
            })
        }
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error processing document: {str(e)}')
        }



