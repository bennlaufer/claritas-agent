import os
import boto3
from dotenv import load_dotenv

load_dotenv()

def get_env_variables():
    return {
        "aws_access_key": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "sf_username": os.getenv("SF_USERNAME"),
        "sf_password": os.getenv("SF_PASSWORD"),
        "sf_security_token": os.getenv("SF_SECURITY_TOKEN"),
        "tavily_api_key": os.getenv('TAVILY_API_KEY')
    }

def create_aws_client(runtime=True):
    service_name = 'bedrock-runtime' if runtime else 'bedrock'
    client = boto3.client(
        service_name=service_name,
        region_name="us-west-2",
    )
    return client
