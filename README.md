

# 🎯 CareerLens
### AI-Powered Job Match System
https://streamable.com/wcn65n  ---- demo vedio link


## 📌 Problem Statement

Job seekers struggle to understand **how well their resume matches a job description** and **what skills they are missing**. Traditional keyword-based matching fails because it misses semantic context — a resume mentioning *"PyTorch"* should match a JD asking for *"deep learning frameworks"*, even without exact keyword overlap.

**CareerLens** solves this using vector similarity search powered by **Endee**, giving candidates:
- ✅ A semantic match score between their resume and any job description
- ✅ Top matching jobs from a pre-indexed database
- ✅ AI-generated skill gap analysis powered by **Groq LLaMA 3.3 70B**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│              Streamlit UI                    │
│   Upload PDF · Paste JD · View Results      │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
Resume Parser          JD Parser
(PyPDF2)               (text input)
    │                     │
    └──────────┬──────────┘
               ▼
      Embedding Engine
  (sentence-transformers
    all-MiniLM-L6-v2 · 384-dim)
               │
               ▼
  ┌────────────────────────────────┐
  │       Endee Vector DB          │
  │  ┌──────────────────────────┐  │
  │  │  resume_index  (1 vec)   │  │
  │  │  jobs_index    (10 vecs) │  │
  │  │  skills_index  (30 vecs) │  │
  │  └──────────────────────────┘  │
  │  cosine similarity · HNSW      │
  └────────────┬───────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
Groq LLM              Results Dashboard
LLaMA 3.3 70B         Match % · Jobs · Gaps
Gap Analysis
```

---

## 🔍 How Endee Is Used

Endee is the **core retrieval engine** powering three features:

| Collection | Purpose | Vectors |
|---|---|---|
| `resume_index` | Stores user's resume as a 384-dim embedding | 1 |
| `jobs_index` | Pre-indexed job postings for semantic job matching | 10 |
| `skills_index` | Individual skill embeddings for gap detection | 30 |

**Key Endee operations used:**
- `create_index()` — create indexes with cosine space type and INT8 precision
- `index.upsert()` — index resume, job, and skill vectors
- `index.query()` — find top-K similar jobs to resume (HNSW search)
- Cosine similarity — computed between resume vector and target JD vector
- Cross-query gap detection — find skills in JD space absent from resume space

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Resume Parser | Extracts and cleans text from any resume PDF |
| 🔍 Semantic Job Matching | Finds best matching jobs using vector similarity — not keywords |
| 📊 Match Score | Cosine similarity score between resume and target JD |
| 🧠 AI Gap Analysis | Groq LLaMA 3.3 70B explains missing skills with actionable advice |
| 🏷️ Skill Gap Tags | Visual badges showing the most important missing skills |
| ⚙️ Configurable | Adjust number of results and minimum match threshold |

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.10+
- Docker Desktop ([download here](https://www.docker.com/products/docker-desktop/))
- Groq API key ([get free key here](https://console.groq.com))

### 1. Clone the repo
```bash
git clone https://github.com/vaasavibokkisam/careerlens.git
cd careerlens
```

### 2. Start Endee vector database
```bash
docker compose up -d
```
Verify at `http://localhost:8080` ✅

### 3. Set up Python environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 4. Configure API keys
```bash
# Create .env file
echo GROQ_API_KEY=your_groq_api_key_here > .env
```

### 5. Run the app
```bash
streamlit run app.py
```
Open `http://localhost:8501` 🚀

---

## 📁 Project Structure

```
careerlens/
├── app.py                  # Streamlit UI — main application
├── docker-compose.yml      # Endee server setup
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
└── utils/
    ├── resume_parser.py    # PDF text extraction (PyPDF2)
    ├── embedder.py         # sentence-transformers wrapper
    ├── endee_client.py     # Endee vector DB client (3 indexes)
    ├── groq_client.py      # Groq LLM — gap analysis + match summary
    └── job_loader.py       # 10 pre-built Bengaluru job listings
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| Vector Database | **Endee** | High recall, low latency, self-hosted |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Free, local, 384-dim semantic vectors |
| LLM | Groq `llama-3.3-70b-versatile` | Fastest LLM API — 750 tokens/sec, free tier |
| UI | Streamlit | Fastest way to build ML demo apps |
| PDF Parsing | PyPDF2 | Simple, lightweight, no API needed |
| Math | NumPy | Cosine similarity computation |
| Server | Docker | Runs Endee as local server in one command |

---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| Cosine similarity | Semantic match score between resume & JD vectors (0–100%) |
| Recall@K | % of relevant jobs retrieved in top-K results |
| Skill gap precision | Relevance of identified missing skills |
| LLM response time | Groq gap analysis latency (~1 second) |

---

## 🔮 Future Improvements

- [ ] Add more job sources via web scraping
- [ ] Resume rewriting suggestions via Groq
- [ ] Multi-language resume support
- [ ] Interview question generation based on skill gaps
- [ ] Evaluation dashboard with recall@K metrics
- [ ] Resume version comparison

---

## 👩‍💻 Built By

**Vaasavi Bokkisam** — ML Intern Project Submission for Endee

---

<div align="center">

Made with ❤️ using Endee · Groq · Streamlit · sentence-transformers

</div>
