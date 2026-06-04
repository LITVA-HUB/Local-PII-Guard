# Integration Guide

## Minimal FastAPI gateway

```python
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

from pii_agent_pro import PIIGateway
from pii_models import PIIGatewayPolicy

app = FastAPI(title="Local PII Guard proxy")
policy = PIIGatewayPolicy(fail_closed=True)
gateway = PIIGateway(model_path="Qwen2.5-3B-Instruct-IQ3_M.gguf", policy=policy)


class TokenizeRequest(BaseModel):
    session_id: str
    text: str


class RestoreRequest(BaseModel):
    session_id: str
    text: str


@app.post("/tokenize")
def tokenize(req: TokenizeRequest):
    return gateway.tokenize(req.session_id, req.text)


@app.post("/restore")
def restore(req: RestoreRequest):
    return {"restored": gateway.restore(req.session_id, req.text)}
```

Run with:

```bash
pip install fastapi uvicorn
uvicorn integration_example:app --reload
```

## Using with OpenAI / Anthropic

```python
import openai
from pii_agent_pro import PIIGateway
from pii_models import PIIGatewayPolicy

policy = PIIGatewayPolicy(fail_closed=True)
gateway = PIIGateway(model_path="Qwen2.5-3B-Instruct-IQ3_M.gguf", policy=policy)

session_id = "user_42"
user_message = "My name is Jane Smith, email jane@example.com. How do I reset my password?"

# 1. Tokenize before sending
result = gateway.tokenize(session_id, user_message)
safe_text = result["tokenized_text"]
# "My name is <<NAME_1>>, email <<EMAIL_1>>. How do I reset my password?"

# 2. Send tokenized text to cloud LLM
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": gateway.cloud_system_prompt},
        {"role": "user", "content": safe_text},
    ],
)

cloud_reply = response.choices[0].message.content
# "Hello <<NAME_1>>, here is how to reset your password..."

# 3. Restore tokens in the response
final_reply = gateway.restore(session_id, cloud_reply)
# "Hello Jane Smith, here is how to reset your password..."
print(final_reply)
```

## Tokenizing structured payloads

```python
payload = {
    "user": "Иван Петров",
    "contacts": {
        "email": "ivan@example.com",
        "phone": "+79991234567",
    },
    "message": "Мой паспорт 4510 123456, карта 4276 1234 5678 9012",
}

safe_payload = gateway.tokenize_payload("session_1", payload)
# {
#   "user": "<<NAME_1>>",
#   "contacts": {"email": "<<EMAIL_1>>", "phone": "<<PHONE_1>>"},
#   "message": "Мой паспорт <<PASSPORT_1>>, карта <<CARD_1>>"
# }

restored = gateway.restore_payload("session_1", safe_payload)
# original payload restored
```

## CRM / database export

```python
fields = gateway.export_fields("session_1")
# {
#   "NAME": ["Jane Smith"],
#   "EMAIL": ["jane@example.com"],
#   "PHONE": ["+79991234567"],
# }

# Write to your CRM or database using the extracted structured values
```

## Session cleanup

```python
# Reset all vault entries for a session when it ends
gateway.vault.reset_session("session_1")
```

## Running without the local LLM model

For deployments where name detection is not needed or the model is unavailable:

```python
from pii_models import PIIGatewayPolicy, DEFAULT_ENABLED_TYPES, ENTITY_NAME

types_without_names = DEFAULT_ENABLED_TYPES - {ENTITY_NAME}
policy = PIIGatewayPolicy(
    enabled_types=types_without_names,
    enable_llm_names=False,
    fail_closed=False,
)
gateway = PIIGateway(policy=policy)
# No model download required. Detects email, phone, card, SNILS, passport, date.
```
