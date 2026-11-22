"""ai_rag_engine.py - AI insights + RAG chatbot with automatic fallback.
Order: Gemini -> Groq -> local model (optional) -> offline summary / document search."""
import glob
import json
import os
import random
import time

from google import genai
from google.genai import errors, types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# ---------- Settings (change models here) ----------
GEMINI_MODELS = ("gemini-3.8-flash",)      # add more names here if our key lists them
GEMINI_SECRET = "My_api_key"
GEMINI_RETRIES = 3

GROQ_MODEL = "openai/gpt-oss-20b"     # change if our Groq model list shows another name
GROQ_SECRET = "Groq_api_key"

USE_LOCAL_MODEL = False                    # True only if you set up the Hugging Face backup
LOCAL_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

last_provider = "none"                     # which AI answered last

SYSTEM_PROMPT = (
    "You are an energy optimization assistant. Answer using ONLY the FORECAST SUMMARY "
    "and the DOCUMENTS provided. If the answer is not in them, say so clearly. "
    "When you use a document, mention its file name in [brackets]. "
    "Be concise, practical and use plain language. Units: demand is in MW."
)


class AIUnavailable(RuntimeError):
    pass


GeminiUnavailable = AIUnavailable  # old name, kept for compatibility


# ---------- Secrets ----------
def get_secret(name):
    """Works in a plain script, in Streamlit and in Colab."""
    key = os.environ.get(name)
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets[name]
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        pass
    raise RuntimeError(f"Secret '{name}' not found.")


# ---------- Provider 1: Gemini ----------
_gemini_client = None


def _ask_gemini(prompt, system):
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=get_secret(GEMINI_SECRET))
    config = types.GenerateContentConfig(system_instruction=system, temperature=0.3)
    last_exc = None
    for model in GEMINI_MODELS:
        for attempt in range(GEMINI_RETRIES):
            try:
                resp = _gemini_client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                if resp.text:
                    return resp.text
                raise RuntimeError("Gemini returned an empty answer")
            except errors.ServerError as exc:      # 500/503: model busy
                last_exc = exc
                time.sleep(min(2 ** attempt, 8) + random.random())
            except errors.ClientError as exc:      # 429 limit, 404 name, 400/403 key
                last_exc = exc
                if getattr(exc, "code", None) == 429:
                    time.sleep(5)
                    continue
                break
    raise last_exc or RuntimeError("Gemini failed")


# ---------- Provider 2: Groq ----------
def _ask_groq(prompt, system):
    from groq import Groq
    client = Groq(api_key=get_secret(GROQ_SECRET))
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ---------- Provider 3: local Hugging Face model (optional) ----------
_local = {}


def _ask_local(prompt, system, max_new_tokens=400):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not _local:
        _local["tok"] = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        _local["model"] = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_ID, torch_dtype="auto", device_map="auto"
        )
    tok, model = _local["tok"], _local["model"]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------- The fallback chain ----------
def ask_llm(prompt, system=SYSTEM_PROMPT):
    global last_provider
    providers = [("Gemini", _ask_gemini), ("Groq", _ask_groq)]
    if USE_LOCAL_MODEL:
        providers.append(("Local model", _ask_local))
    for name, fn in providers:
        try:
            text = fn(prompt, system)
            last_provider = name
            return text
        except Exception as exc:
            print(f"[info] {name} unavailable: {type(exc).__name__}: {str(exc)[:120]}")
    last_provider = "none"
    raise AIUnavailable("All AI providers are busy right now. Please try again in a minute.")


ask_gemini = ask_llm  # old name, kept for compatibility


# ---------- Data from Day 1 ----------
def load_summary():
    for name in ("forecast_summary.json", "summary.json"):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("No summary JSON in data/. Run forecast_engine.py first.")


# ---------- RAG: document search ----------
def chunk_text(text, max_chars=700):
    chunks, current = [], ""
    for part in text.split("\n\n"):
        if current and len(current) + len(part) > max_chars:
            chunks.append(current.strip())
            current = ""
        current += part + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


class DocIndex:
    """Reads .md/.txt files from docs/ and finds the most relevant chunks."""

    def __init__(self, docs_dir=DOCS_DIR):
        self.chunks = []  # list of (source_file, text)
        paths = glob.glob(os.path.join(docs_dir, "*.md")) + glob.glob(os.path.join(docs_dir, "*.txt"))
        for path in sorted(paths):
            with open(path, encoding="utf-8") as f:
                for chunk in chunk_text(f.read()):
                    self.chunks.append((os.path.basename(path), chunk))
        self.vectorizer = None
        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform([c[1] for c in self.chunks])

    def search(self, query, k=3):
        if not self.vectorizer:
            return []
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = sims.argsort()[::-1][:k]
        return [(self.chunks[i][0], self.chunks[i][1], float(sims[i])) for i in best if sims[i] > 0.05]


def format_context(hits):
    if not hits:
        return "(no relevant documents found)"
    return "\n\n".join(f"[{src}]\n{text}" for src, text, _ in hits)


# ---------- Offline fallbacks (no AI needed) ----------
def fallback_insights(summary):
    g = summary.get
    return (
        "The AI service is busy, so this is an automatic summary of the forecast numbers.\n\n"
        f"- Forecast window: {g('forecast_start')} to {g('forecast_end')}\n"
        f"- Average demand: {g('avg_predicted_mw')} MW\n"
        f"- Peak: {g('peak_load_mw')} MW at {g('peak_load_timestamp')}\n"
        f"- Lowest: {g('lowest_load_mw')} MW at {g('lowest_load_timestamp')}\n"
        f"- Simulated peak-shaving effect: {g('estimated_mwh_saved')} MWh\n"
        f"- Validation error (MAPE): {g('validation_mape_percent')}%\n"
    )


def fallback_answer(hits):
    if not hits:
        return "The AI service is busy and no matching document passage was found. Please try again in a minute."
    text = "\n\n".join(f"[{src}]\n{chunk}" for src, chunk, _ in hits)
    return "The AI service is busy, so here are the most relevant passages from our documents:\n\n" + text


# ---------- AI insights ----------
def generate_insights(summary, index):
    hits = index.search("peak demand tariff load shifting maintenance low demand", k=3)
    prompt = f"""FORECAST SUMMARY (JSON):
{json.dumps(summary, indent=2)}

DOCUMENTS:
{format_context(hits)}

Write a short insight report (max 250 words) with these sections:
1. Overview: what the next 10 days look like.
2. Peak risk: when the peak happens and how high it is.
3. Recommended actions: 3 practical energy optimization actions based on the documents.
4. Caveat: mention the validation MAPE and the data source, and say that the savings
   figures come from a simulated peak-shaving scenario, not measured savings.
"""
    try:
        return ask_llm(prompt)
    except AIUnavailable:
        return fallback_insights(summary)


# ---------- RAG chatbot ----------
def answer_question(question, summary, index, history=None):
    """history: list of {"role": "user" | "assistant", "content": str}."""
    hits = index.search(question, k=3)
    chat_history = ""
    for turn in (history or [])[-6:]:
        chat_history += f"{turn['role'].upper()}: {turn['content']}\n"
    prompt = f"""FORECAST SUMMARY (JSON):
{json.dumps(summary, indent=2)}

DOCUMENTS:
{format_context(hits)}

CONVERSATION SO FAR:
{chat_history or '(none)'}

QUESTION: {question}
"""
    try:
        return ask_llm(prompt), hits
    except AIUnavailable:
        return fallback_answer(hits), hits


if __name__ == "__main__":
    summary = load_summary()
    index = DocIndex()
    print(generate_insights(summary, index))
    print("\nAnswered by:", last_provider)
