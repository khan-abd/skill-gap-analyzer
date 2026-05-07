"""
resume_parser.py
Extracts Name, CGPA, and Skills from a PDF resume.
Uses: pypdf (text extraction) + regex + keyword matching.
No external AI API required — fully offline.
"""

import re
import io
import pypdf

# ── Master skill vocabulary (same as generate_data.py + extras from Kaggle) ──
SKILL_VOCAB = [
    # Programming
    "Python", "SQL", "R", "Java", "Scala", "C++", "C", "Go",
    "JavaScript", "TypeScript", "MATLAB", "VBA", "Bash", "SAS", "SPSS",
    # Data & Analytics
    "Excel", "PowerBI", "Power BI", "Tableau", "Looker", "Qlik",
    "Google Analytics", "MicroStrategy", "Alteryx",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-Learn",
    "EDA", "ETL", "Statistics", "Data Visualization", "Data Analysis",
    "Spreadsheet", "BigQuery",
    # Machine Learning
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Keras", "XGBoost", "Scikit-Learn",
    "Regression", "Clustering", "Classification", "Time Series",
    "Forecasting", "Hypothesis Testing", "A/B Testing",
    # Databases
    "MySQL", "PostgreSQL", "MongoDB", "NoSQL", "Snowflake",
    "SQL Server", "Oracle", "Redis",
    # Cloud & DevOps
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Git", "GitHub",
    # Product & Business
    "Product Roadmap", "Product Management", "Agile", "Scrum",
    "JIRA", "User Research", "Figma", "Go-To-Market", "Market Research",
    "Business Analysis", "Requirements Gathering", "Stakeholder Management",
    "Financial Modeling", "Consulting Frameworks", "Deck Building",
    "Strategic Thinking", "Communication", "Leadership",
    # Web & APIs
    "APIs", "REST", "JSON", "Web Scraping", "BeautifulSoup", "Requests",
    "Flask", "Django", "FastAPI", "Streamlit",
    # Marketing
    "SEO", "CRM", "Salesforce", "PowerPoint",
    # Other
    "Hadoop", "Spark", "Kafka", "Hive",
]

# Build a lowercase → canonical mapping for fast lookup
_SKILL_MAP = {s.lower(): s for s in SKILL_VOCAB}
# Also handle common abbreviations
_EXTRA_ALIASES = {
    "power bi":    "PowerBI",
    "power_bi":    "PowerBI",
    "powerbi":     "PowerBI",
    "power_bi":    "PowerBI",
    "ml":          "Machine Learning",
    "dl":          "Deep Learning",
    "sklearn":     "Scikit-Learn",
    "scikit":      "Scikit-Learn",
    "tf":          "TensorFlow",
    "ms excel":    "Excel",
    "ms office":   "Excel",
    "google sheets": "Excel",
    "nlp":         "NLP",
    "cv":          "Computer Vision",
    "tableau":     "Tableau",
    "powerpoint":  "PowerPoint",
    "ms powerpoint": "PowerPoint",
}
_SKILL_MAP.update(_EXTRA_ALIASES)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Read all pages of a PDF and return raw text."""
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_name(text: str) -> str:
    """
    Heuristic: the name is usually the very first non-empty line of a resume.
    We skip lines that look like phone numbers, emails, or URLs.
    """
    skip_patterns = re.compile(
        r"(http|www\.|linkedin|github|@|\+\d|\d{10}|resume|curriculum)", re.I
    )
    for line in text.splitlines():
        line = line.strip()
        if line and not skip_patterns.search(line) and len(line.split()) <= 5:
            # Likely a name if it's short and has no special chars (except spaces)
            if re.match(r"^[A-Za-z\s\.]+$", line):
                return line
    return ""


def extract_cgpa(text: str) -> float | None:
    """
    Search for CGPA / GPA / CPI patterns like:
      CGPA: 7.67   |   GPA 3.8/4.0   |   CPI: 8.9   |   7.67 CGPA
    Returns a float or None if not found.
    """
    patterns = [
        r"(?:cgpa|gpa|cpi)[:\s\-]*([0-9]\.[0-9]{1,2})",       # CGPA: 7.67
        r"([0-9]\.[0-9]{1,2})\s*(?:cgpa|gpa|cpi)",             # 7.67 CGPA
        r"([0-9]\.[0-9]{1,2})\s*/\s*(?:10|4)",                 # 7.67/10 or 3.8/4
        # Fallback: decimal in 5.0–9.9 range near education keywords
        r"(?:b\.?tech|engineering|university|college|grade)[^\n]*?([5-9]\.[0-9]{1,2})",
        # Last resort: first standalone decimal in 5.0–9.99 range on a line
        r"^[^\n]*?([5-9]\.[0-9]{1,2})[^\n]*$",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE | re.MULTILINE):
            try:
                val = float(m.group(1))
            except (IndexError, ValueError):
                continue
            # Normalize /4 scale to /10
            if val <= 4.0:
                val = round(val * 2.5, 2)
            # Must be a plausible GPA on 10-pt scale
            if 5.0 <= val <= 10.0:
                return round(val, 2)
    return None


def extract_skills(text: str) -> list[str]:
    """
    Scan the resume text for any skill from SKILL_VOCAB.
    Uses whole-word matching to avoid false positives (e.g., 'R' inside words).
    Returns a deduplicated, sorted list of matched canonical skill names.
    """
    text_lower = text.lower()
    found: set[str] = set()

    for kw, canonical in _SKILL_MAP.items():
        # Use word-boundary regex so "R" doesn't match inside "Research"
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            found.add(canonical)

    return sorted(found)


def parse_resume(pdf_bytes: bytes) -> dict:
    """
    Main entry point. Returns a dict with:
      - name    : str
      - cgpa    : float | None
      - skills  : list[str]
      - raw_text: str  (for debugging)
    """
    text  = extract_text_from_pdf(pdf_bytes)
    name  = extract_name(text)
    cgpa  = extract_cgpa(text)
    skills = extract_skills(text)

    return {
        "name":     name,
        "cgpa":     cgpa,
        "skills":   skills,
        "raw_text": text,
    }
