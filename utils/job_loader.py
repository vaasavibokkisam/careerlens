from utils.embedder import get_embeddings_batch

SAMPLE_JOBS = [
    {
        "id": "job_001",
        "title": "Machine Learning Engineer",
        "company": "DataCore AI",
        "location": "Bengaluru, India",
        "description": (
            "We are looking for an ML Engineer to build and deploy production ML models. "
            "Strong Python, PyTorch/TensorFlow, MLOps, and cloud experience required. "
            "You will work on recommendation systems, NLP pipelines, and model serving."
        ),
    },
    {
        "id": "job_002",
        "title": "AI Research Intern",
        "company": "DeepMind",
        "location": "Bengaluru, India (Remote)",
        "description": (
            "Join our research team working on large language models, reinforcement learning, "
            "and neural architecture search. Strong Python, PyTorch, and research paper reading skills needed."
        ),
    },
    {
        "id": "job_003",
        "title": "Data Scientist",
        "company": "Flipkart",
        "location": "Bengaluru, India",
        "description": (
            "Analyse large datasets, build predictive models, and deliver business insights. "
            "Experience with SQL, Python, scikit-learn, and data visualisation required. "
            "Work on pricing, fraud detection, and supply chain optimisation."
        ),
    },
    {
        "id": "job_004",
        "title": "NLP Engineer",
        "company": "Sarvam AI",
        "location": "Bengaluru, India",
        "description": (
            "Build multilingual NLP systems for Indian languages. "
            "Experience with transformers, HuggingFace, fine-tuning LLMs, and text classification required. "
            "Work on speech-to-text, translation, and dialogue systems."
        ),
    },
    {
        "id": "job_005",
        "title": "MLOps Engineer",
        "company": "Walmart Labs",
        "location": "Bengaluru, India",
        "description": (
            "Design and maintain ML infrastructure including model training pipelines, "
            "feature stores, and model monitoring. Experience with Kubernetes, Docker, "
            "Airflow, and cloud platforms (AWS/GCP) required."
        ),
    },
    {
        "id": "job_006",
        "title": "Computer Vision Engineer",
        "company": "Niramai",
        "location": "Bengaluru, India",
        "description": (
            "Develop deep learning models for medical image analysis. "
            "Strong Python, OpenCV, PyTorch, and CNN architecture experience required. "
            "Work on object detection, segmentation, and anomaly detection."
        ),
    },
    {
        "id": "job_007",
        "title": "AI Product Intern",
        "company": "Zomato",
        "location": "Bengaluru, India",
        "description": (
            "Work at the intersection of AI and product. Help build recommendation engines, "
            "search ranking systems, and personalisation features. "
            "Python, SQL, and basic ML knowledge required."
        ),
    },
    {
        "id": "job_008",
        "title": "Generative AI Engineer",
        "company": "Infosys AI Lab",
        "location": "Bengaluru, India",
        "description": (
            "Build enterprise GenAI applications using LLMs, RAG pipelines, and vector databases. "
            "Experience with LangChain, OpenAI API, embeddings, and prompt engineering required."
        ),
    },
    {
        "id": "job_009",
        "title": "Backend Engineer (AI Platform)",
        "company": "Swiggy",
        "location": "Bengaluru, India",
        "description": (
            "Build scalable backend services for AI-powered features. "
            "Strong Python/Go, REST APIs, microservices, and database experience. "
            "Work on real-time ML inference infrastructure."
        ),
    },
    {
        "id": "job_010",
        "title": "Research Engineer — Vector Search",
        "company": "Endee",
        "location": "Bengaluru, India",
        "description": (
            "Join the team building Endee, a high-performance vector database. "
            "Work on embedding pipelines, ANN indexing, RAG systems, and semantic search. "
            "Python, C++, and systems programming experience valued."
        ),
    },
]

SAMPLE_SKILLS = [
    "Python", "PyTorch", "TensorFlow", "scikit-learn", "machine learning",
    "deep learning", "NLP", "transformers", "HuggingFace", "LangChain",
    "RAG pipelines", "vector databases", "embeddings", "SQL", "Docker",
    "Kubernetes", "MLOps", "cloud AWS GCP", "computer vision", "OpenCV",
    "data analysis", "pandas", "numpy", "REST API", "FastAPI",
    "reinforcement learning", "generative AI", "LLMs", "fine-tuning",
    "prompt engineering",
]

_jobs_loaded = False


def load_sample_jobs(db):
    """Embed and index sample jobs + skills into Endee. Runs only once per session."""
    global _jobs_loaded
    if _jobs_loaded:
        return

    # Embed job descriptions
    descriptions = [j["description"] for j in SAMPLE_JOBS]
    vectors = get_embeddings_batch(descriptions)

    jobs_with_vectors = []
    for job, vec in zip(SAMPLE_JOBS, vectors):
        jobs_with_vectors.append({**job, "vector": vec})

    db.upsert_jobs(jobs_with_vectors)

    # Embed skill labels
    skill_vectors = get_embeddings_batch(SAMPLE_SKILLS)
    skill_records = [
        {
            "id": f"skill_{i:03d}",
            "vector": vec,
            "meta": {"skill": skill},
        }
        for i, (skill, vec) in enumerate(zip(SAMPLE_SKILLS, skill_vectors))
    ]
    db.upsert_skills(skill_records)

    _jobs_loaded = True