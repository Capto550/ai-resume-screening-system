from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st

from src.extract import DEFAULT_SKILLS, extract_highlights
from src.parsing import extract_text_from_upload
from src.preprocess import normalize_text
from src.scoring import score_resumes_against_jd


st.set_page_config(page_title="AI Resume Screening", layout="wide")

st.title("AI Resume Screening System")
st.caption(
    "Upload resumes (PDF/TXT) and a job description to get a TF‑IDF similarity match score with explainable top terms."
)


def _bytes_from_uploaded_file(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    data = uploaded_file.getvalue()
    return data if isinstance(data, bytes) else bytes(data)


with st.sidebar:
    st.subheader("Job description")
    jd_text_input = st.text_area("Paste job description text", height=220, placeholder="Paste JD here...")
    jd_file = st.file_uploader("Or upload JD (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=False)

    st.subheader("Resumes")
    resumes_files = st.file_uploader(
        "Upload resumes (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    st.subheader("Options")
    min_score = st.slider("Minimum match score", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    top_terms_k = st.slider("Top matched terms to show", min_value=3, max_value=15, value=8, step=1)
    use_default_skills = st.checkbox("Extract skills (heuristic)", value=True)
    run_btn = st.button("Run screening", type="primary")


def build_job_description_text() -> tuple[str, list[str]]:
    parts: list[str] = []
    sources: list[str] = []
    if jd_text_input and jd_text_input.strip():
        parts.append(jd_text_input.strip())
        sources.append("pasted_text")
    if jd_file is not None:
        jd_bytes = _bytes_from_uploaded_file(jd_file)
        jd_text = extract_text_from_upload(jd_file.name, jd_bytes)
        if jd_text.strip():
            parts.append(jd_text.strip())
            sources.append(f"uploaded_file:{jd_file.name}")
        else:
            sources.append(f"uploaded_file_empty:{jd_file.name}")
    return ("\n\n".join(parts).strip(), sources)


def parse_resumes() -> tuple[list[tuple[str, str]], dict[str, Any]]:
    parsed: list[tuple[str, str]] = []
    meta: dict[str, Any] = {"errors": [], "empty": []}
    for f in resumes_files or []:
        b = _bytes_from_uploaded_file(f)
        txt = extract_text_from_upload(f.name, b)
        if not txt.strip():
            meta["empty"].append(f.name)
            continue
        parsed.append((f.name, txt))
    return parsed, meta


if run_btn:
    jd_raw, jd_sources = build_job_description_text()
    resumes_raw, resume_meta = parse_resumes()

    if not jd_raw.strip():
        st.error("Job description is empty after parsing. Please paste text and/or upload a readable PDF/TXT.")
        st.stop()
    if not resumes_raw:
        st.error("No resumes with readable text found. Please upload at least one readable PDF/TXT resume.")
        if resume_meta.get("empty"):
            st.info("Empty/failed parses: " + ", ".join(resume_meta["empty"]))
        st.stop()

    if resume_meta.get("empty"):
        st.warning("Some resumes produced no text: " + ", ".join(resume_meta["empty"]))

    jd_clean = normalize_text(jd_raw)
    resumes_clean = [(fn, normalize_text(t)) for fn, t in resumes_raw]

    scores = score_resumes_against_jd(jd_clean, resumes_clean, top_k_terms=top_terms_k)
    max_score = max((s.score for s in scores), default=0.0)
    scale = 1.0 / max_score if max_score > 0 else 1.0

    highlights_map = {}
    if use_default_skills:
        for fn, raw_text in resumes_raw:
            highlights_map[fn] = extract_highlights(raw_text, skills=DEFAULT_SKILLS)

    rows: list[dict[str, Any]] = []
    for rank, s in enumerate(scores, start=1):
        norm_score = s.score * scale
        if norm_score < min_score:
            continue
        hl = highlights_map.get(s.filename)
        rows.append(
            {
                "rank": rank,
                "filename": s.filename,
                "match_score": round(norm_score, 4),
                "top_matched_terms": ", ".join(s.top_terms),
                "skills_found": ", ".join(hl.skills) if hl else "",
                "years_experience_heuristic": hl.years_experience if hl else None,
                "education_heuristic": hl.education if hl else None,
            }
        )

    st.subheader("Results")
    st.caption(
        "Scores are TF‑IDF cosine similarities normalized so the best-matching resume is 1.0. "
        "Skills/education/years are best-effort heuristics."
    )
    #st.write("JD:", jd_clean[:])
    #st.write("RESUME:", resumes_clean[0][1][:300])
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No resumes meet the current minimum score filter.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="resume_screening_results.csv",
            mime="text/csv",
        )

    with st.expander("Job description (parsed)", expanded=False):
        st.write(f"Sources: {', '.join(jd_sources) if jd_sources else 'none'}")
        st.text_area("Parsed JD text", value=jd_raw, height=240)

else:
    st.info("Add a job description and resumes in the sidebar, then click **Run screening**.")

