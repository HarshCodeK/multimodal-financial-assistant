# Multimodal Financial Assistant

![Project Screenshot](../image.png)

## What this does

Upload a screenshot or PDF of a credit card statement, receipt, or invoice and ask "Why was this charge deducted?" The system reads the document, extracts the key fields (vendor, line items, totals, date), finds relevant policies from a local knowledge base, and generates a grounded answer explaining the charge.

## Why RAG + vision extraction together

Vision reads the document so you don't have to type anything in. RAG grounds the explanation in real policy text (refund rules, subscription billing terms, late fee policies) so the LLM doesn't just make up a reason — it cites actual rules from the knowledge base.

## Architecture

```
Upload (PDF/Image)
      |
      v
Parse (PyMuPDF for PDF, path for image)
      |
      v
Vision/Text Extract (Groq LLM — vision model for images, text model for PDFs)
      |
      v
Structured fields (vendor, items, totals, flagged_charge)
      |
      v
Retrieve Policy Context (SentenceTransformer embeddings + ChromaDB)
      |
      v
LLM (Groq llama-3.3-70b) combines fields + policy + question
      |
      v
Grounded Answer
```

## How to run

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Build the policy knowledge base (run once):
   ```
   python -c "from src.knowledge_base import build_knowledge_base; build_knowledge_base()"
   ```
4. Launch the UI:
   ```
   streamlit run app.py
   ```
5. Upload a financial document and ask a question.

## Worked example

**Upload:** `sample (1).jpg` — a credit card statement screenshot.

**Extracted fields:**
```json
{
  "vendor": "Unknown",
  "line_items": [
    {"description": "Spotify Subscription", "amount": 11.99},
    {"description": "Progressive Insurance", "amount": 145.50},
    {"description": "Target Purchase", "amount": 88.20},
    {"description": "Grand Hotel (2 nights)", "amount": 340.00},
    {"description": "Amazon.com Purchase", "amount": 55.00}
  ],
  "subtotal": 947.68,
  "tax": 24.50,
  "total": 4512.78,
  "date": "12/05/2026",
  "flagged_charge": {"description": "Progressive Insurance", "amount": 145.50}
}
```

**Question:** "Why was this charge deducted?"

**Answer:** The Progressive Insurance charge of $145.50 was deducted as part of a recurring subscription billing cycle. Per policy, charges are processed on the same calendar day each month as the original sign-up date. Users receive an email notification 48 hours before each recurring charge.

**Policy sources used:** Subscription billing policy (recurring charges, billing cycle), refund policy (timeframes), late fee policy (when fees apply).

## What I'd add next

- **Docling** for better table extraction from complex PDF layouts
- **Multi-statement comparison** to detect duplicate charges or billing changes over time
- **Confidence scoring** on extracted fields so the user knows which fields the model is unsure about
- **Streaming responses** for faster UX feedback
- **Support for handwritten receipts** with higher-resolution vision models
