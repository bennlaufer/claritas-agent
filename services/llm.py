import json
from config.env import create_aws_client

bedrock_runtime = create_aws_client()

def invoke_model(body, model_id, accept, content_type):
    try:
        response = bedrock_runtime.invoke_model(
            body=json.dumps(body),
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )
        return response
    except Exception as e:
        print(f"Error invoking {model_id}: {e}")
        raise e
