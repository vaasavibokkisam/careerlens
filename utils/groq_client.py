import os
from groq import Groq

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        _client = Groq(api_key=api_key)
    return _client


def _call(system: str, user: str, max_tokens: int = 600) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # updated model
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def explain_gaps(resume_text: str, jd_text: str) -> str:
    """Use Groq to explain skill gaps between resume and JD."""
    system = (
        "You are a professional career coach. Analyse the resume and job description "
        "provided. Identify the most important skill gaps concisely. Use bullet points. "
        "Be specific and actionable. Keep it under 200 words."
    )
    user = (
        f"RESUME (excerpt):\n{resume_text[:1500]}\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:1500]}\n\n"
        "What are the key skill gaps and how can the candidate address them?"
    )
    return _call(system, user)


def summarize_match(resume_text: str, jd_text: str, score: float) -> str:
    """Generate a one-paragraph match summary."""
    pct = int(score * 100)
    system = (
        "You are a concise career advisor. In 2-3 sentences, summarise how well "
        "a candidate's resume matches a job description. Be honest and constructive."
    )
    user = (
        f"The vector similarity score is {pct}%.\n\n"
        f"RESUME (excerpt):\n{resume_text[:800]}\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:800]}\n\n"
        "Provide a brief match summary."
    )
    return _call(system, user, max_tokens=200)