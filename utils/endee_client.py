import numpy as np
from endee import Endee, Precision
from endee.exceptions import ConflictException


class EndeeClient:
    """
    Wrapper around the Endee vector database SDK.

    Indexes used:
      - resume_index  : stores the user's resume embedding
      - jobs_index    : stores job postings (pre-indexed)
      - skills_index  : stores skill embeddings for gap analysis
    """

    DIMENSION = 384  # all-MiniLM-L6-v2 output dimension

    def __init__(self):
        self.client = Endee()

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _vec(id: str, vector: list, meta: dict) -> dict:
        """Build a vector record with all required fields including filter."""
        return {
            "id":     id,
            "vector": vector,
            "meta":   meta,
            "filter": {},   # required by SDK even if empty
        }

    # ── Setup ──────────────────────────────────────────────────────────────

    def setup_collections(self):
        """Create indexes if they don't already exist."""
        for name in ["resume_index", "jobs_index", "skills_index"]:
            try:
                self.client.create_index(
                    name=name,
                    dimension=self.DIMENSION,
                    space_type="cosine",
                    precision=Precision.INT8,
                )
            except ConflictException:
                pass  # index already exists, skip

    # ── Resume ─────────────────────────────────────────────────────────────

    def upsert_resume(self, vector: list[float], text: str):
        index = self.client.get_index("resume_index")
        index.upsert([
            self._vec("user_resume", vector, {"text": text[:500]})
        ])

    # ── Jobs ───────────────────────────────────────────────────────────────

    def upsert_jobs(self, jobs: list[dict]):
        index   = self.client.get_index("jobs_index")
        records = [
            self._vec(
                job["id"],
                job["vector"],
                {
                    "title":    job["title"],
                    "company":  job["company"],
                    "location": job["location"],
                    "snippet":  job["description"][:300],
                },
            )
            for job in jobs
        ]
        index.upsert(records)

    def search_jobs(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        index   = self.client.get_index("jobs_index")
        results = index.query(vector=query_vector, top_k=top_k)

        matches = []
        for r in results:
            rid   = r["id"]         if isinstance(r, dict) else r.id
            score = r["similarity"] if isinstance(r, dict) else r.similarity
            meta  = r["meta"]       if isinstance(r, dict) else getattr(r, "meta", {})
            if not meta:
                meta = {}
            matches.append({
                "id":       rid,
                "score":    score,
                "title":    meta.get("title",    ""),
                "company":  meta.get("company",  ""),
                "location": meta.get("location", ""),
                "snippet":  meta.get("snippet",  ""),
            })
        return matches

    # ── Skills ────────────────────────────────────────────────────────────

    def upsert_skills(self, skills: list[dict]):
        index   = self.client.get_index("skills_index")
        records = [
            self._vec(
                s["id"],
                s["vector"],
                s.get("meta", {}),
            )
            for s in skills
        ]
        index.upsert(records)

    def find_skill_gaps(
        self,
        resume_vec: list[float],
        jd_vec:     list[float],
        threshold:  float = 0.45,
    ) -> list[str]:
        skills_index  = self.client.get_index("skills_index")
        jd_skills     = skills_index.query(vector=jd_vec,     top_k=15)
        resume_skills = skills_index.query(vector=resume_vec, top_k=15)

        def _id(r):
            return r["id"] if isinstance(r, dict) else r.id

        def _sim(r):
            return r["similarity"] if isinstance(r, dict) else r.similarity

        def _meta(r):
            m = r["meta"] if isinstance(r, dict) else getattr(r, "meta", {})
            return m if m else {}

        resume_ids = {_id(r) for r in resume_skills if _sim(r) > threshold}

        gaps = [
            _meta(r).get("skill", _id(r))
            for r in jd_skills
            if _id(r) not in resume_ids and _sim(r) > threshold
        ]
        return gaps[:8]

    # ── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def cosine_score(vec_a: list[float], vec_b: list[float]) -> float:
        a     = np.array(vec_a)
        b     = np.array(vec_b)
        score = float(np.dot(a, b))
        return max(0.0, min(1.0, score))