import json
import numpy as np
import pandas as pd
from pathlib import Path
from langchain.tools import Tool
from config.env import create_aws_client

aws_client = create_aws_client()

# --- read embeddings ---
base_path = Path(__file__).resolve().parent.parent
pickle_path = base_path / 'data' / 'PRIZM_Embedded.pkl'
dft = pd.read_pickle(pickle_path)

def embed_documents_with_cohere(texts):
    if isinstance(texts, str):
        texts = [texts]
        single = True
    else:
        single = False

    model_id = "cohere.embed-english-v3"
    input_type = "clustering"
    truncate = "NONE"
    json_params = {
        'texts': [t[:2048] for t in texts],
        'truncate': truncate,
        'input_type': input_type
    }
    result = aws_client.invoke_model(
        body=json.dumps(json_params),
        modelId=model_id
    )
    response = json.loads(result['body'].read().decode())
    embeddings = [np.array(vec) for vec in response['embeddings']]
    return embeddings[0] if single else embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_prizm_segments(user_prompt):
    query_vector = embed_documents_with_cohere(user_prompt)
    results = []
    for idx, row in dft.iterrows():
        score = cosine_similarity(row["embedding"], query_vector)
        results.append((idx, score))
    results.sort(key=lambda x: x[1], reverse=True)
    top_matches = []
    for i, (idx, score) in enumerate(results[:3]):
        segment = dft.iloc[idx]["prizm_segment"]
        description = dft.iloc[idx]["text"]
        top_matches.append(
            f"Match #{i+1}:\nSegment: {segment}\nDescription: {description}\nRelevance Score: {round(score, 3)}\n"
        )
    return "\n".join(top_matches) or "No matches found."

def create_rag_tool():
    return Tool(
        name="prizm_segment_matcher",
        func=search_prizm_segments,
        description="Use this tool to match a prompt to PRIZM segments using semantic search."
    )
