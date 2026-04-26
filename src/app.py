# src/app.py – T&C Risk Scanner with Fairness Score & Risk Level
import streamlit as st
import time
from pypdf import PdfReader

from retriever import embed, retrieve
from inference import rag_answer

st.set_page_config(page_title="T&C Risk Scanner", layout="wide")

# -----------------------------
# DEBUG (set to False for final)
# -----------------------------
DEBUG = False

def debug(msg, value=None):
    if DEBUG:
        st.write(f"🧪 {msg}")
        if value is not None:
            st.write(value)

# -----------------------------
# PDF HELPER
# -----------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

# -----------------------------
# ENHANCED RISK SCORING (negation + severity)
# -----------------------------
RISK_WEIGHTS = {
    "arbitration": 5,
    "class action": 5,
    "waive": 5,
    "indemnify": 5,
    "without notice": 5,
    "liability": 2,
    "not responsible": 4,
    "disclaim": 3,
    "terminate": 3,
    "sole discretion": 4,
    "close account": 3,
    "share your data": 3,
    "disclosure": 2,
    "auto-renew": 2,
    "restrict": 2,
    "own risk": 2,
    "cannot guarantee": 2,
}

NEGATION_WORDS = {"not", "no", "never", "without", "exclude", "except", "doesn't", "does not"}

def compute_risk_score(context: str) -> int:
    """Return integer risk score between 0 and 100 (higher = more risk)."""
    context_lower = context.lower()
    sentences = context_lower.split(". ")
    total_risk = 0
    max_risk_per_sentence = 10

    for sent in sentences:
        sent_risk = 0
        has_negation = any(neg in sent for neg in NEGATION_WORDS)
        for keyword, weight in RISK_WEIGHTS.items():
            if keyword in sent:
                if has_negation:
                    sent_risk += weight * .35      # negation reduces impact
                else:
                    sent_risk += weight
        total_risk += min(sent_risk, max_risk_per_sentence)

    num_sentences = max(1, len(sentences))
    avg_risk_per_sentence = total_risk / num_sentences
    raw_score = min(100, avg_risk_per_sentence * 10)
    return int(raw_score)

def fairness_score_from_risk(risk: int) -> int:
    return 100 - risk

def risk_level(risk: int) -> str:
    if risk < 15:
        return "Low"
    elif risk < 25:
        return "Moderate"
    else:
        return "High"

def compute_context_coverage(context: str) -> tuple:
    """Return (accuracy, recall) based on risk category presence."""
    categories = ["liability", "data", "arbitration", "termination", "indemnity"]
    found = []
    for cat in categories:
        appears = cat in context.lower()
        has_neg = any(neg in context.lower() for neg in NEGATION_WORDS)
        if appears and not has_neg:
            found.append(1)      # clearly present
        elif appears and has_neg:
            found.append(0.5)    # negated
        else:
            found.append(0)      # absent
    # accuracy = proportion fully present
    acc = sum(1 for f in found if f == 1) / len(categories)
    # recall = proportion at least partially present (including negated)
    rec = sum(1 for f in found if f > 0) / len(categories)
    return acc, rec

# -----------------------------
# CHUNKING (LIMITED FOR SPEED)
# -----------------------------
def chunk_text(text, max_chunk_size=800):
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) < max_chunk_size:
            current += " " + line
        else:
            if current.strip():
                chunks.append(current.strip())
            current = line
    if current.strip():
        chunks.append(current.strip())
    return chunks[:10]   # limit for performance

# -----------------------------
# CACHE EMBEDDINGS
# -----------------------------
@st.cache_data
def cached_embeddings(chunks):
    return embed(chunks)

# -----------------------------
# MAIN UI
# -----------------------------
st.title("📄 Terms & Conditions Risk Scanner")
st.warning(
    "⚠️ This tool is not legal advice. It uses heuristic risk scoring and a lightweight model. "
    "Always consult a qualified legal professional for legal decisions."
)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
pasted_text = st.text_area("Or paste Terms & Conditions text", height=200)

tc_text = None

if pasted_text.strip():
    tc_text = pasted_text.strip()
elif uploaded_file:
    with st.spinner("Extracting PDF..."):
        tc_text = extract_text_from_pdf(uploaded_file)

if tc_text:
    st.success(f"Loaded text ({len(tc_text)} characters)")

    chunks = chunk_text(tc_text)

    if not chunks:
        st.error("Text too short or not formatted.")
        st.stop()

    st.info(f"Using {len(chunks)} chunks (limited for speed)")

    if st.button("🔍 Analyze Document"):
        start_total = time.time()

        # Step 1: Embeddings
        with st.spinner("Computing embeddings..."):
            chunk_embeddings = cached_embeddings(chunks)

        # Step 2: Retrieve best matching chunk
        query = "What are the main legal risks in this document?"
        top_chunks = retrieve(query, chunks, chunk_embeddings, top_k=1)
        context = top_chunks[0] if top_chunks else ""

        if not context:
            st.error("No relevant context found.")
            st.stop()

        # Step 3: Generate model answer (optional, for display)
        with st.spinner("Generating analysis (few seconds)..."):
            try:
                model_answer = rag_answer(query, context)
            except Exception as e:
                model_answer = f"Model error: {e}"

        # Step 4: Compute enhanced metrics
        risk = compute_risk_score(context)
        fairness = fairness_score_from_risk(risk)
        level = risk_level(risk)
        accuracy, recall = compute_context_coverage(context)

        # Display results
        st.subheader("📊 Risk Analysis")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fairness Score", f"{fairness}/100")
        col2.metric("Risk Level", level)
        col3.metric("Precision (Risk Coverage)", f"{accuracy:.2f}")
        col4.metric("Recall (Risk Detection)", f"{recall:.2f}")

        with st.expander("🔍 Retrieved context used for scoring"):
            st.write(context)

        debug("Total pipeline time", f"{time.time() - start_total:.2f}s")

else:
    st.info("Upload a PDF or paste text to begin.")