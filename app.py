import os
import re
import json
import datetime
import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# =========================
# PASTE YOUR REAL API KEYS HERE
# =========================
OPENAI_API_KEY = "YOUR REAL KEY "
XAI_API_KEY = "YOUR REAL KEY"

# =========================
# STORAGE SETUP
# =========================
STORAGE_DIR = "storage/runs"
os.makedirs(STORAGE_DIR, exist_ok=True)

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Output Risk & Reliability Checker",
    layout="wide"
)

SYSTEM = (
    "You are an evidence-critical reviewer of biomedical/scientific documents. "
    "Be cautious and highlight uncertainty. Never diagnose or recommend treatment. "
    "Advisory-only."
)

# =========================
# CLIENTS
# =========================
openai_client = OpenAI(api_key=OPENAI_API_KEY)
grok_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# =========================
# HELPERS
# =========================
def pdf_to_text(file_bytes, max_pages=10):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    for i in range(min(len(doc), max_pages)):
        chunks.append(doc[i].get_text("text"))
    return "\n".join(chunks)

def normalize_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def trim_to_char_limit(text, limit=18000):
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.3):]
    return head + "\n\n...[TRUNCATED]...\n\n" + tail

def build_prompt(doc_text):
    return f"""
Return valid JSON only (no markdown).

Keys:
- summary (5-8 bullets)
- assumptions (5-12)
- weaknesses (5-12)
- open_questions (5-12)
- notes (3-6)

DOCUMENT:
\"\"\"{doc_text}\"\"\"
""".strip()

def extract_json_anyway(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON found in model output.")
        return json.loads(match.group(0))

def call_gpt(model, prompt):
    res = openai_client.responses.create(
        model=model,
        instructions=SYSTEM,
        input=prompt
    )
    return res.output_text

def call_grok(model, prompt):
    res = grok_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return res.output_text

def normalize_list(items):
    cleaned = []
    for s in items:
        s = re.sub(r"^\-+\s*", "", s.strip().lower())
        s = re.sub(r"\s+", " ", s)
        cleaned.append(s)
    return cleaned

def divergence(a, b):
    A, B = set(normalize_list(a)), set(normalize_list(b))
    return {
        "overlap": sorted(A & B),
        "only_a": sorted(A - B),
        "only_b": sorted(B - A),
        "overlap_count": len(A & B),
        "only_a_count": len(A - B),
        "only_b_count": len(B - A),
    }

def save_run(filename, gpt_out, grok_out, div_w, div_q, gpt_model, grok_model):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    data = {
        "run_id": ts,
        "filename": filename,
        "timestamp": datetime.datetime.now().isoformat(),
        "models": {
            "gpt": gpt_model,
            "grok": grok_model
        },
        "divergence": {
            "weaknesses": {
                "gpt_only": div_w["only_a_count"],
                "grok_only": div_w["only_b_count"],
                "overlap": div_w["overlap_count"]
            },
            "open_questions": {
                "gpt_only": div_q["only_a_count"],
                "grok_only": div_q["only_b_count"],
                "overlap": div_q["overlap_count"]
            }
        },
        "outputs": {
            "gpt": gpt_out,
            "grok": grok_out
        }
    }

    path = os.path.join(STORAGE_DIR, f"{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path, data

def list_saved_runs():
    if not os.path.exists(STORAGE_DIR):
        return []
    files = [f for f in os.listdir(STORAGE_DIR) if f.endswith(".json")]
    files.sort(reverse=True)
    return files

# =========================
# UI
# =========================
st.title("AI Output Risk & Reliability Checker (UI)")
st.caption("Advisory-only. Not diagnosis, not treatment, not correctness scoring.")

with st.expander("⚠️ Disclaimer", expanded=True):
    st.write(
        "This tool highlights uncertainty and differences between models for human review only. "
        "It does not provide clinical advice or correctness judgments."
    )

uploaded = st.file_uploader("Upload a biomedical paper (PDF)", type=["pdf"])
colA, colB = st.columns(2)

with st.sidebar:
    st.header("Models")
    GPT_MODEL = st.text_input("GPT model", value="gpt-4.1-mini")
    GROK_MODEL = st.text_input("Grok model", value="grok-4")
    max_pages = st.slider("Pages to extract", 1, 15, 10)
    run_btn = st.button("Analyze")

    st.markdown("---")
    st.subheader("Previous Runs")
    previous_runs = list_saved_runs()
    if previous_runs:
        for f in previous_runs[:5]:
            st.write(f)
    else:
        st.caption("No saved runs yet.")

if run_btn and uploaded:
    try:
        bytes_data = uploaded.read()
        text = pdf_to_text(bytes_data, max_pages=max_pages)
        text = trim_to_char_limit(normalize_text(text))

        prompt = build_prompt(text)

        with st.spinner("Calling GPT & Grok..."):
            gpt_raw = call_gpt(GPT_MODEL, prompt)
            grok_raw = call_grok(GROK_MODEL, prompt)

        gpt_out = extract_json_anyway(gpt_raw)
        grok_out = extract_json_anyway(grok_raw)

        div_w = divergence(gpt_out["weaknesses"], grok_out["weaknesses"])
        div_q = divergence(gpt_out["open_questions"], grok_out["open_questions"])

        saved_file, saved_data = save_run(
            uploaded.name,
            gpt_out,
            grok_out,
            div_w,
            div_q,
            GPT_MODEL,
            GROK_MODEL
        )

        st.success(f"Saved run: {saved_file}")

        with colA:
            st.subheader("GPT Output")
            st.json(gpt_out)

        with colB:
            st.subheader("Grok Output")
            st.json(grok_out)

        st.markdown("---")
        st.subheader("Epistemic Divergence (Highlights)")

        c1, c2, c3 = st.columns(3)
        c1.metric("Unique weaknesses (GPT)", div_w["only_a_count"])
        c2.metric("Unique weaknesses (Grok)", div_w["only_b_count"])
        c3.metric("Overlap weaknesses", div_w["overlap_count"])

        st.write("**Only GPT saw:**", div_w["only_a"][:10])
        st.write("**Only Grok saw:**", div_w["only_b"][:10])

        st.markdown("**Open questions divergence**")
        st.write("GPT-only:", div_q["only_a"][:10])
        st.write("Grok-only:", div_q["only_b"][:10])

        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(saved_data, indent=2, ensure_ascii=False),
            file_name=f"{saved_data['run_id']}_analysis.json",
            mime="application/json"
        )

    except Exception as e:
        st.error(f"Error during analysis: {e}")

else:
    st.info("Upload a PDF and click Analyze.")