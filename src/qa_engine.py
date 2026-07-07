import json
from groq import Groq
from dotenv import load_dotenv
from src.vision_extractor import extract_fields
from src.knowledge_base import retrieve_policy_context

load_dotenv()

def answer_question(document: dict, question: str) -> dict:
    extracted = extract_fields(document)
    if "error" in extracted:
        return {"answer": "Could not extract fields from the document.", "extracted_fields": extracted, "policy_sources_used": []}

    vendor = extracted.get("vendor", "")
    flagged = extracted.get("flagged_charge", "")
    if isinstance(flagged, dict):
        flagged_desc = flagged.get("description", "")
    else:
        flagged_desc = str(flagged)

    search_query = f"{vendor} {flagged_desc}".strip() if vendor else flagged_desc
    policy_chunks = retrieve_policy_context(search_query)

    fields_text = json.dumps(extracted, indent=2)
    policy_text = "\n\n".join(policy_chunks)

    prompt = f"""You are a financial assistant. Use the extracted document fields and the policy context below to answer the user's question.

EXTRACTED FIELDS:
{fields_text}

POLICY CONTEXT:
{policy_text}

USER QUESTION: {question}

Provide a clear, grounded answer referencing specific policy details where relevant."""

    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "extracted_fields": extracted,
        "policy_sources_used": policy_chunks,
    }
