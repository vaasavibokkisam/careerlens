import streamlit as st
import os
from utils.resume_parser import parse_resume
from utils.embedder import get_embedding
from utils.endee_client import EndeeClient
from utils.groq_client import explain_gaps, summarize_match
from utils.job_loader import load_sample_jobs
import json

st.set_page_config(
    page_title="CareerLens — AI Job Match",
    page_icon="🎯",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'DM Serif Display', serif; }
  .match-card {
    background: #f8f9ff;
    border: 1px solid #e0e4ff;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 12px;
  }
  .score-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 13px;
    font-weight: 600;
  }
  .score-high { background: #d4f7e4; color: #1a6b3c; }
  .score-mid  { background: #fff3cd; color: #7a5900; }
  .score-low  { background: #fde8e8; color: #8b1a1a; }
  .gap-tag {
    display: inline-block;
    background: #fff0e6;
    color: #b34700;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    margin: 3px 3px 3px 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Title ───────────────────────────────────────────────────────────────────
st.title("🎯 CareerLens")
st.markdown("**Upload your resume · Paste a job description · Discover your best matches & skill gaps**")
st.divider()

# ── Initialize Endee ────────────────────────────────────────────────────────
@st.cache_resource
def init_endee():
    client = EndeeClient()
    client.setup_collections()
    load_sample_jobs(client)
    return client

with st.spinner("Initialising vector database..."):
    db = init_endee()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of job matches", 1, 10, 5)
    score_threshold = st.slider("Minimum match score (%)", 0, 100, 30)
    st.divider()
    st.caption("Powered by Endee · Groq · sentence-transformers")

# ── Main columns ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📄 Your Resume")
    uploaded = st.file_uploader("Upload PDF resume", type=["pdf"])
    resume_text = ""
    if uploaded:
        with st.spinner("Parsing resume..."):
            resume_text = parse_resume(uploaded)
        st.success(f"Resume parsed — {len(resume_text.split())} words extracted")
        with st.expander("Preview extracted text"):
            st.text(resume_text[:1200] + ("..." if len(resume_text) > 1200 else ""))

with col2:
    st.subheader("📋 Target Job Description")
    jd_text = st.text_area(
        "Paste the job description here",
        height=220,
        placeholder="Paste any job description you're interested in..."
    )

st.divider()

# ── Analyse button ────────────────────────────────────────────────────────────
if st.button("🔍 Analyse & Find Matches", type="primary", use_container_width=True):
    if not resume_text:
        st.warning("Please upload your resume first.")
    elif not jd_text.strip():
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Embedding your resume..."):
            resume_vec = get_embedding(resume_text)
            db.upsert_resume(resume_vec, resume_text)

        with st.spinner("Finding best matching jobs from database..."):
            job_matches = db.search_jobs(resume_vec, top_k=top_k)

        with st.spinner("Scoring against your target JD..."):
            jd_vec = get_embedding(jd_text)
            jd_score = db.cosine_score(resume_vec, jd_vec)

        with st.spinner("Analysing skill gaps with Groq AI..."):
            gap_analysis = explain_gaps(resume_text, jd_text)
            summary = summarize_match(resume_text, jd_text, jd_score)

        # ── Results ──────────────────────────────────────────────────────────
        st.subheader("📊 Results")

        tab1, tab2, tab3 = st.tabs(["🎯 JD Match", "💼 Top Job Matches", "📈 Skill Gap Analysis"])

        with tab1:
            pct = int(jd_score * 100)
            pill_class = "score-high" if pct >= 70 else ("score-mid" if pct >= 40 else "score-low")
            emoji = "🟢" if pct >= 70 else ("🟡" if pct >= 40 else "🔴")
            st.markdown(f"""
            <div class="match-card">
              <h3 style="margin:0 0 8px">Your resume vs target JD</h3>
              <span class="score-pill {pill_class}">{emoji} {pct}% Match</span>
              <p style="margin-top:14px;color:#444">{summary}</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            if job_matches:
                for job in job_matches:
                    score_pct = int(job["score"] * 100)
                    if score_pct < score_threshold:
                        continue
                    pill = "score-high" if score_pct >= 70 else ("score-mid" if score_pct >= 40 else "score-low")
                    st.markdown(f"""
                    <div class="match-card">
                      <b>{job['title']}</b> &nbsp;
                      <span class="score-pill {pill}">{score_pct}% match</span><br>
                      <span style="color:#666;font-size:13px">{job['company']} · {job['location']}</span>
                      <p style="margin:8px 0 0;font-size:13px;color:#444">{job['snippet']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No matches found above threshold. Try lowering the minimum score.")

        with tab3:
            st.markdown("#### 🧠 AI-Powered Gap Analysis")
            st.markdown(gap_analysis)

            st.markdown("#### 🔖 Missing Skills (quick tags)")
            gaps = db.find_skill_gaps(resume_vec, jd_vec)
            if gaps:
                tags_html = "".join([f'<span class="gap-tag">⚡ {g}</span>' for g in gaps])
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.success("No major skill gaps detected!")
