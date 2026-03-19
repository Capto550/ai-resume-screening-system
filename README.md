# AI Resume Screening System (Streamlit)

Upload candidate resumes (PDF/TXT) and a job description (paste and/or upload). The app extracts text, computes a TF‑IDF cosine-similarity match score, and shows a ranked list with explainable top matched terms + CSV export.

## Setup (Windows PowerShell)

```powershell
cd "c:\Users\Nisha\OneDrive\Desktop\resume_screening"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

## Notes
- Everything runs in-memory per session (no database).
- Match scores are guidance for screening; they should not be used as the sole decision signal.

