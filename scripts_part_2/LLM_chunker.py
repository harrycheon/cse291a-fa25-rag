from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_article_semantically(article: str, model="gpt-4.1-mini") -> list[str]:
    prompt = f"""
    Split this financial article into semantic topical chunks.

    RULES:
    - Each chunk must be a contiguous excerpt from the original text
    - DO NOT paraphrase
    - DO NOT hallucinate text
    - Use 2–10 chunks max
    - Output only JSON in schema format:
      {{
         "chunks": ["<original excerpt>", "<original excerpt>", ...]
      }}

    ARTICLE:
    \"\"\"{article}\"\"\"
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SemanticChunks",
                "schema": {
                    "type": "object",
                    "properties": {
                        "chunks": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["chunks"],
                    "additionalProperties": False
                }
            },
        }
    )

    # Parse structured JSON from the assistant message
    structured = json.loads(response.choices[0].message.content)
    return structured["chunks"]
