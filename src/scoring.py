from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ResumeScore:
    filename: str
    score: float
    top_terms: list[str]


def _top_terms_for_resume(
    vectorizer: TfidfVectorizer,
    jd_vec,
    resume_vec,
    top_k: int,
) -> list[str]:
    # Contribution proxy: elementwise product of JD and resume tf-idf weights.
    try:
        contrib = jd_vec.multiply(resume_vec)
        if contrib.nnz == 0:
            return []
        idx = contrib.indices
        data = contrib.data
        order = np.argsort(data)[::-1][:top_k]
        feature_names = vectorizer.get_feature_names_out()
        return [str(feature_names[int(idx[i])]) for i in order]
    except Exception:
        return []


def score_resumes_against_jd(
    jd_text: str,
    resumes: Iterable[tuple[str, str]],
    *,
    top_k_terms: int = 8,
) -> list[ResumeScore]:
    resume_list = list(resumes)
    docs = [jd_text] + [t for _, t in resume_list]
    vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    lowercase=True,
    strip_accents="unicode"
)
    tfidf = vectorizer.fit_transform(docs)
    jd_vec = tfidf[0]
    resume_vecs = tfidf[1:]

    sims = cosine_similarity(resume_vecs, jd_vec).reshape(-1)
    results: list[ResumeScore] = []
    for (filename, _), score, rvec in zip(resume_list, sims, resume_vecs):
        top_terms = _top_terms_for_resume(vectorizer, jd_vec, rvec, top_k_terms)
        results.append(
            ResumeScore(
                filename=filename,
                score=float(score),
                top_terms=top_terms,
            )
        )
    print("JD vector sum:", jd_vec.sum())
    print("Resume vector sum:", resume_vecs[0].sum())
    results.sort(key=lambda r: r.score, reverse=True)
    return results

