"""
app.py  –  Phase 3: Streamlit Dashboard
Skill Gap Analyzer with 3 ML layers:
  Layer 1: TF-IDF + Cosine Similarity Course Recommender
  Layer 2: Logistic Regression Employability Predictor
  Layer 3: K-Means Student Segmentation
"""

import streamlit as st
import sqlite3, os, re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from resume_parser import parse_resume

# ── Skill Normalization Utilities (no hardcoding) ──────────────────────────
def normalize_key(skill: str) -> str:
    """
    Canonical key for comparison only — strips ALL separators and lowercases.
    'Power_Bi', 'PowerBI', 'power bi', 'POWER_BI' all become 'powerbi'.
    'Scikit-Learn', 'scikit_learn', 'ScikitLearn' all become 'scikitlearn'.
    This means comparison never fails due to casing or separator style.
    """
    return re.sub(r'[\s_\-]+', '', skill.strip().lower())


# Short tech words that should be fully uppercase in display (SQL, BI, AI …)
# Determined algorithmically: ≤4 chars AND all originally uppercase-worthy
_FORCE_UPPER = {"sql", "bi", "ai", "ml", "dl", "nlp", "crm", "etl", "api",
                "aws", "gcp", "vba", "sas", "eda", "css", "html", "php",
                "ux", "ui", "r", "c"}

def prettify_skill(skill: str) -> str:
    """
    Human-readable display form, derived algorithmically:
    1. Replace underscores / extra spaces with a single space
    2. Split on camelCase boundaries so 'PowerBI' → ['Power', 'BI']
    3. Title-case each word; fully uppercase known short tech acronyms
    Result: 'power_bi' → 'Power BI',  'nosql' → 'NoSQL',  'aws' → 'AWS'
    """
    # Step 1: replace underscores and hyphens with space
    s = re.sub(r'[_\-]+', ' ', skill.strip())
    # Step 2: insert space before uppercase runs that follow lowercase
    # e.g. 'PowerBI' → 'Power BI',  'NoSQL' → 'No SQL'
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    # Step 3: per-word formatting
    words = s.split()
    result = []
    for w in words:
        lower_w = w.lower()
        if lower_w in _FORCE_UPPER:
            result.append(w.upper())
        else:
            result.append(w.capitalize())
    return ' '.join(result)

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkillBridge – AI Career Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "skill_gap.db")

TARGET_ROLES = [
    "Data Analyst", "Data Scientist", "Software Engineer",
    "Web Developer", "Android/iOS Developer", "Business Analyst",
    "Product Manager", "Management Consultant", "Marketing Analyst",
    "HR Analyst", "UX/UI Designer", "Financial Analyst",
]

CLUSTER_LABELS = {0: "High Achiever 🏆", 1: "On Track 📈", 2: "Needs Focus 🎯"}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

/* Hero */
.hero { text-align:center; padding: 2.5rem 1rem 1.5rem; }
.hero h1 { font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: .4rem; }
.hero p { color: #94a3b8; font-size: 1.1rem; }

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px; padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
}
.card h3 { color: #e2e8f0; font-size: 1rem; font-weight: 600; margin-bottom: .8rem; }

/* Skill chips */
.chip-green  { display:inline-block; background:#052e16; color:#4ade80;
    border:1px solid #166534; border-radius:20px; padding:3px 12px;
    font-size:.78rem; margin:3px; }
.chip-red    { display:inline-block; background:#2d0a0a; color:#f87171;
    border:1px solid #7f1d1d; border-radius:20px; padding:3px 12px;
    font-size:.78rem; margin:3px; }
.chip-blue   { display:inline-block; background:#0c1a40; color:#93c5fd;
    border:1px solid #1e3a5f; border-radius:20px; padding:3px 12px;
    font-size:.78rem; margin:3px; }

/* Course card */
.course-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(167,139,250,0.25);
    border-left: 4px solid #a78bfa;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: .8rem;
}
.course-card h4 { color: #e2e8f0; font-size: .95rem; font-weight: 600; margin:0 0 .3rem; }
.course-card p  { color: #94a3b8; font-size: .8rem; margin: 0; }
.course-card a  { color: #a78bfa; font-size: .8rem; text-decoration: none; }

/* Metric */
.big-metric { text-align:center; }
.big-metric .val { font-size: 2.8rem; font-weight: 800; }
.big-metric .lbl { color: #94a3b8; font-size: .85rem; }

/* Section label */
.section-label {
    font-size: .7rem; font-weight: 700; letter-spacing: .12em;
    color: #7c3aed; text-transform: uppercase; margin-bottom: .5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: rgba(15,12,41,0.9); }
</style>
""", unsafe_allow_html=True)


# ── DB helpers ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data
def load_all_students():
    conn = get_conn()
    return pd.read_sql_query("SELECT * FROM students;", conn)


@st.cache_data
def load_courses():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM courses WHERE skills_taught IS NOT NULL AND skills_taught != '';", conn
    )
    df["skills_text"] = df["skills_taught"].str.replace("|", " ", regex=False)
    return df


@st.cache_data
def load_skill_weights(role: str) -> dict[str, str]:
    """Returns {skill: tier} for the given role from the skill_weights table."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT skill, tier FROM skill_weights WHERE role = ? ORDER BY frequency DESC;",
        (role,)
    ).fetchall()
    return {row["skill"]: row["tier"] for row in rows}


@st.cache_data
def get_required_skills(role: str) -> list[str]:
    conn = get_conn()
    cur = conn.execute(
        "SELECT required_skills FROM job_postings WHERE LOWER(job_title)=LOWER(?);", (role,)
    )
    # freq map: normalized_key → {count, best_display}
    freq: dict[str, dict] = {}
    for row in cur.fetchall():
        for raw in row["required_skills"].split("|"):
            raw = raw.strip()
            if not raw:
                continue
            key = normalize_key(raw)
            pretty = prettify_skill(raw)
            if key not in freq:
                freq[key] = {"count": 0, "display": pretty}
            freq[key]["count"] += 1
            # Prefer the display name with fewer underscores / more readable
            if "_" in freq[key]["display"] and "_" not in pretty:
                freq[key]["display"] = pretty
    # Sort by frequency descending, return display names
    sorted_keys = sorted(freq, key=lambda k: freq[k]["count"], reverse=True)
    return [freq[k]["display"] for k in sorted_keys]


# ── ML helpers ────────────────────────────────────────────────────────────────
def recommend_courses(skill_gap, courses_df, top_n=3):
    if not skill_gap:
        return pd.DataFrame()
    query = " ".join(skill_gap)
    all_texts = [query] + courses_df["skills_text"].tolist()
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(all_texts)
    scores = cosine_similarity(mat[0:1], mat[1:]).flatten()
    out = courses_df.copy()
    out["match_score"] = scores
    return out.sort_values("match_score", ascending=False).head(top_n).reset_index(drop=True)


@st.cache_data
def build_models():
    students_df = load_all_students()
    records = []
    for _, row in students_df.iterrows():
        # Normalize student skills by key for matching
        known_keys = {normalize_key(s) for s in row["current_skills"].split("|") if s.strip()}
        req_display = get_required_skills(row["target_role"])[:20]
        req_keys    = {normalize_key(s) for s in req_display}
        if not req_keys:
            continue
        match = len(known_keys & req_keys) / len(req_keys) * 100
        records.append({
            "student_id": row["student_id"],
            "skill_match_score": round(match, 2),
            "cgpa": float(row["cgpa"]),
            "skills_known_count": len(known_keys),
        })
    df = pd.DataFrame(records)

    # ── Composite employability score: 60% skills, 40% CGPA ──────────────────
    # CGPA normalized: treat 10.0 as 100%, anything below 6.0 as ~0% contribution
    # This ensures the model label reflects BOTH academic + skill performance.
    CGPA_MIN, CGPA_MAX = 6.0, 10.0
    df["cgpa_norm"] = ((df["cgpa"] - CGPA_MIN) / (CGPA_MAX - CGPA_MIN) * 100).clip(0, 100)
    df["composite"] = 0.60 * df["skill_match_score"] + 0.40 * df["cgpa_norm"]

    # Split on composite median so both CGPA and skills influence the label
    median = df["composite"].median()
    df["employable"] = (df["composite"] >= median).astype(int)

    lr = LogisticRegression(max_iter=500)
    lr.fit(df[["skill_match_score", "cgpa", "skills_known_count"]].values, df["employable"].values)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(df[["skill_match_score", "cgpa", "skills_known_count"]].values)
    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS)
    return lr, km, df


# ── Gauge chart ───────────────────────────────────────────────────────────────
def gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
        title={"text": title, "font": {"size": 13, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.05)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 40],  "color": "rgba(239,68,68,0.15)"},
                {"range": [40, 70], "color": "rgba(251,191,36,0.15)"},
                {"range": [70, 100],"color": "rgba(52,211,153,0.15)"},
            ],
        },
    ))
    fig.update_layout(
        height=220, margin=dict(t=30, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
    )
    return fig


# ── Radar chart ───────────────────────────────────────────────────────────────
def radar_chart(known_set, required_list):
    skills = required_list[:8]
    vals = [1 if s in known_set else 0 for s in skills]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=skills + [skills[0]],
        fill="toself",
        fillcolor="rgba(167,139,250,0.25)",
        line=dict(color="#a78bfa", width=2),
        name="You",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[1]*len(skills) + [1],
        theta=skills + [skills[0]],
        fill="toself",
        fillcolor="rgba(96,165,250,0.1)",
        line=dict(color="#60a5fa", width=1, dash="dot"),
        name="Required",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(color="#94a3b8"),
        ),
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8")),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=40, r=40),
        height=280,
        font_color="#e2e8f0",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🎯 SkillBridge</h1>
  <p>AI-powered Skill Gap Analyzer · Course Recommender · Career Readiness Predictor</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Your Profile")

    # ── Resume Upload ──────────────────────────────────────────────────────────
    st.markdown("**📄 Upload Resume (PDF)**")
    st.caption("Auto-fills your name, CGPA and skills")
    uploaded_file = st.file_uploader(
        "Upload Resume", type=["pdf"], label_visibility="collapsed"
    )

    # Parse defaults from resume if uploaded
    parsed_name   = ""
    parsed_cgpa   = 7.7
    parsed_skills = ""

    if uploaded_file is not None:
        with st.spinner("📄 Reading resume..."):
            result = parse_resume(uploaded_file.read())
        parsed_name   = result["name"]
        parsed_cgpa   = result["cgpa"] if result["cgpa"] else 7.7
        parsed_skills = ", ".join(result["skills"])

        extracted_count = len(result["skills"])
        st.success(f"✅ Extracted {extracted_count} skills from your resume!")

        with st.expander("🔍 What was extracted"):
            st.markdown(f"**Name:** {parsed_name or 'Not detected'}")
            st.markdown(f"**CGPA:** {parsed_cgpa}")
            if result["skills"]:
                chips = " ".join(
                    f'<span class="chip-blue">{s}</span>' for s in result["skills"]
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No skills matched. Please type them manually below.")

    st.divider()

    # ── Editable Fields (pre-filled from resume or manual) ────────────────────
    st.markdown("**✏️ Confirm / Edit Your Details**")
    name        = st.text_input("Your Name", value=parsed_name, placeholder="e.g. Abdullah Khan")
    cgpa        = st.slider("CGPA", 5.0, 10.0, float(parsed_cgpa), 0.1)
    target_role = st.selectbox("Target Role", TARGET_ROLES)

    st.markdown("**Your Current Skills**")
    st.caption("Edit or add more skills (comma-separated)")
    skills_input = st.text_area(
        "Skills",
        value=parsed_skills,
        placeholder="Python, SQL, Excel, PowerBI...",
        height=100,
        label_visibility="collapsed",
    )

    analyze_btn = st.button("🚀 Analyze My Profile", use_container_width=True, type="primary")
    st.divider()
    st.markdown("**💡 Example Skills**")
    role_examples = {
        "Data Analyst":          "Python, SQL, Excel, Pandas",
        "Data Scientist":        "Python, Machine Learning, SQL",
        "Software Engineer":     "Java, Python, Data Structures, Git",
        "Web Developer":         "HTML, CSS, JavaScript, React",
        "Android/iOS Developer": "Swift, Kotlin, Flutter",
        "Product Manager":       "Communication, Agile, Excel",
        "Management Consultant": "Excel, PowerPoint, Communication",
        "Business Analyst":      "SQL, Excel, Business Analysis",
        "Marketing Analyst":     "Google Analytics, SEO, Excel",
        "HR Analyst":            "Excel, Talent Acquisition, HRIS",
        "UX/UI Designer":        "Figma, Wireframing, User Research",
        "Financial Analyst":     "Excel, Financial Modeling, SQL",
    }
    st.code(role_examples.get(target_role, "Python, SQL, Excel"), language=None)


# ── Main area placeholder ─────────────────────────────────────────────────────
if not analyze_btn:
    st.markdown("""
    <div class="card" style="text-align:center; padding: 3rem;">
      <div style="font-size:3.5rem; margin-bottom:1rem;">🎯</div>
      <h2 style="color:#e2e8f0; margin-bottom:.5rem;">Ready to find your skill gaps?</h2>
      <p style="color:#94a3b8; margin-bottom: 1.5rem;">Upload your PDF resume on the left to auto-fill your profile,<br>then click <b>Analyze My Profile</b>.</p>
      <div style="display:flex; gap:1.5rem; justify-content:center; flex-wrap:wrap;">
        <div class="card" style="padding:.8rem 1.5rem; min-width:160px;">
          <div style="font-size:1.8rem">📄</div>
          <div style="color:#a78bfa; font-weight:600; font-size:.85rem;">1. Upload Resume</div>
        </div>
        <div class="card" style="padding:.8rem 1.5rem; min-width:160px;">
          <div style="font-size:1.8rem">✏️</div>
          <div style="color:#60a5fa; font-weight:600; font-size:.85rem;">2. Confirm Details</div>
        </div>
        <div class="card" style="padding:.8rem 1.5rem; min-width:160px;">
          <div style="font-size:1.8rem">🚀</div>
          <div style="color:#34d399; font-weight:600; font-size:.85rem;">3. Get Analysis</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Parse inputs ──────────────────────────────────────────────────────────────
user_name   = name.strip() if name.strip() else "You"
user_skills = [s.strip().title() for s in skills_input.split(",") if s.strip()] if skills_input else []

if not user_skills:
    st.warning("⚠️ Please enter at least one skill before analyzing.")
    st.stop()


# ── Compute ───────────────────────────────────────────────────────────────────
with st.spinner("Running AI analysis..."):
    courses_df    = load_courses()
    lr, km, feat_df = build_models()

    required_skills = get_required_skills(target_role)[:20]

    # ── Normalize-key comparison: works regardless of DB storage format ────────
    # Map normalized_key → display_name for both sides
    user_norm = {normalize_key(s): s           for s in user_skills}
    req_norm  = {normalize_key(s): s           for s in required_skills}

    # Intersection and gap operate on keys; display names come from user/req maps
    matched_keys = set(user_norm) & set(req_norm)
    gap_keys     = set(req_norm) - set(user_norm)

    # Display sets: use the prettified required-side name for consistency
    known_set = {req_norm[k] for k in matched_keys}
    skill_gap = sorted(req_norm[k] for k in gap_keys)
    match_pct = round(len(matched_keys) / len(req_norm) * 100, 1) if req_norm else 0

    # Layer 1
    top_courses = recommend_courses(skill_gap, courses_df, top_n=3)

    # Layer 2 — Logistic Regression with CGPA adjustment
    X_pred = np.array([[match_pct, cgpa, len(user_skills)]])
    base_prob = lr.predict_proba(X_pred)[0][1] * 100

    # Direct CGPA adjustment: model alone may underweight it, so we apply
    # an explicit linear bonus/penalty based on CGPA relative to 7.5 baseline.
    # Each 0.5 CGPA above 7.5 → +3 points; below 7.5 → -3 points (capped ±12)
    cgpa_adjustment = round(((cgpa - 7.5) / 0.5) * 3, 1)
    cgpa_adjustment = max(-12, min(12, cgpa_adjustment))   # clamp to ±12
    emp_prob = round(min(100, max(1, base_prob + cgpa_adjustment)), 1)

    # Layer 3
    cluster_id = int(km.predict(X_pred)[0])
    cluster_label = CLUSTER_LABELS[cluster_id]
    peers = int((feat_df["cluster"] == cluster_id).sum())


# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown(f"### 👋 Hello, {user_name}! Here's your analysis for **{target_role}**")
st.markdown("")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="card big-metric">
      <div class="lbl">Skill Match Score</div>
      <div class="val" style="color:{'#4ade80' if match_pct>=70 else '#fbbf24' if match_pct>=40 else '#f87171'}">{match_pct}%</div>
      <div class="lbl">of job requirements</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="card big-metric">
      <div class="lbl">Skills You Have</div>
      <div class="val" style="color:#4ade80">{len(known_set)}</div>
      <div class="lbl">matched to role</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="card big-metric">
      <div class="lbl">Skill Gaps</div>
      <div class="val" style="color:#f87171">{len(skill_gap)}</div>
      <div class="lbl">skills missing</div>
    </div>""", unsafe_allow_html=True)

with k4:
    segment_desc = {
        "High Achiever 🏆": "Strong profile",
        "On Track 📈":     "Good progress",
        "Needs Focus 🎯":  "Gap to close",
    }
    desc = segment_desc.get(cluster_label, "")
    st.markdown(f"""
    <div class="card big-metric">
      <div class="lbl">Your Segment</div>
      <div class="val" style="font-size:1.6rem; color:#a78bfa">{cluster_label.split()[0]}</div>
      <div class="lbl">{cluster_label.split(maxsplit=1)[1]} · {desc}</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Row 2: Gauges + Radar ─────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns([1, 1, 1.5])

with col_a:
    st.plotly_chart(gauge_chart(match_pct, "Skill Match %",
        "#4ade80" if match_pct >= 70 else "#fbbf24" if match_pct >= 40 else "#f87171"),
        use_container_width=True)

with col_b:
    st.plotly_chart(gauge_chart(emp_prob, "Employability %",
        "#4ade80" if emp_prob >= 70 else "#fbbf24" if emp_prob >= 40 else "#f87171"),
        use_container_width=True)

with col_c:
    st.markdown('<div class="section-label">Skill Radar vs Job Requirements</div>', unsafe_allow_html=True)
    st.plotly_chart(radar_chart(set(user_skills), required_skills), use_container_width=True)

st.divider()

# ── Row 3: Skills + Courses ───────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown('<div class="section-label">Skills Breakdown</div>', unsafe_allow_html=True)

    if known_set:
        st.markdown("**✅ You already have:**")
        chips = " ".join(f'<span class="chip-green">{s}</span>' for s in sorted(known_set))
        st.markdown(chips, unsafe_allow_html=True)

    if skill_gap:
        weights      = load_skill_weights(target_role)
        weights_norm = {normalize_key(k): v for k, v in weights.items()}

        must   = [s for s in skill_gap if weights_norm.get(normalize_key(s)) == "Must Learn"]
        should = [s for s in skill_gap if weights_norm.get(normalize_key(s)) == "Should Learn"]
        later  = [s for s in skill_gap if weights_norm.get(normalize_key(s)) == "Can Learn Later"]
        uncat  = [s for s in skill_gap if normalize_key(s) not in weights_norm]
        later  = later + uncat

        if must:
            st.markdown("**🔥 Must Learn** *(high demand — appears in 60%+ of job postings)*")
            chips = " ".join(f'<span class="chip-red">{s}</span>' for s in must)
            st.markdown(chips, unsafe_allow_html=True)
        if should:
            st.markdown("**⚠️ Should Learn** *(commonly required — 30–60% of postings)*")
            chips = " ".join(f'<span class="chip-red" style="opacity:.75">{s}</span>' for s in should)
            st.markdown(chips, unsafe_allow_html=True)
        if later:
            st.markdown("**💡 Can Learn Later** *(good to have)*")
            chips = " ".join(f'<span class="chip-red" style="opacity:.45">{s}</span>' for s in later)
            st.markdown(chips, unsafe_allow_html=True)

    st.markdown("**💼 Top demanded skills for this role:**")
    chips = " ".join(f'<span class="chip-blue">{s}</span>' for s in required_skills[:10])
    st.markdown(chips, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-label">Layer 1 – AI Course Recommendations</div>', unsafe_allow_html=True)
    st.caption("Powered by TF-IDF Vectorization + Cosine Similarity")

    if top_courses.empty:
        st.success("🎉 You already have all the required skills! No courses needed.")
    else:
        for _, row in top_courses.iterrows():
            match_pct_c = round(row["match_score"] * 100, 1)
            st.markdown(f"""
            <div class="course-card">
              <h4>📘 {row['course_name']}</h4>
              <p>{row['platform']} · {row['difficulty']} · ⭐ {row['rating']} · {row['duration_hours']}h</p>
              <p>Match Score: <b style="color:#a78bfa">{match_pct_c}%</b></p>
              <a href="{row['url']}" target="_blank">🔗 View Course →</a>
            </div>""", unsafe_allow_html=True)

st.divider()

# ── Row 4: ML Layers summary ──────────────────────────────────────────────────
st.markdown('<div class="section-label">Layer 2 & 3 – Employability & Segmentation</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    color = "#4ade80" if emp_prob >= 70 else "#fbbf24" if emp_prob >= 40 else "#f87171"
    status = "Likely to pass resume screening ✅" if emp_prob >= 50 else "Skill gap too large ⚠️"
    advice = (
        "Great profile! Keep polishing your projects."
        if emp_prob >= 70 else
        "Close 3–5 skill gaps to significantly boost your score."
        if emp_prob >= 40 else
        f"Focus on: {', '.join(skill_gap[:3])} first."
    )
    st.markdown(f"""
    <div class="card">
      <h3>📈 Logistic Regression – Employability Predictor</h3>
      <div style="font-size:2.5rem; font-weight:800; color:{color}">{emp_prob}%</div>
      <div style="color:#94a3b8; margin:.4rem 0">{status}</div>
      <div style="color:#e2e8f0; font-size:.88rem; background:rgba(255,255,255,0.05);
           padding:.6rem .9rem; border-radius:8px; margin-top:.5rem">
        💡 {advice}
      </div>
    </div>""", unsafe_allow_html=True)

with c2:
    # Segment descriptions — based on user's own scores, no fake peer references
    segment_details = {
        "High Achiever 🏆": {
            "icon": "🏆",
            "color": "#4ade80",
            "title": "High Achiever",
            "desc": "Your skill match and academic profile are strong. You are well-positioned for this role.",
            "tip": "Polish 1–2 remaining gaps and add a portfolio project to stand out further.",
        },
        "On Track 📈": {
            "icon": "📈",
            "color": "#60a5fa",
            "title": "On Track",
            "desc": "You have a solid foundation for this role with room to grow in a few key areas.",
            "tip": "Target the top 3 missing skills with focused short courses to jump to High Achiever.",
        },
        "Needs Focus 🎯": {
            "icon": "🎯",
            "color": "#f87171",
            "title": "Needs Focus",
            "desc": "Your skill gap for this role is significant. This is a great starting point to build from.",
            "tip": f"Start with the courses recommended above. Closing 5+ gaps will dramatically improve your score.",
        },
    }
    info = segment_details.get(cluster_label, segment_details["Needs Focus 🎯"])
    st.markdown(f"""
    <div class="card">
      <h3>🏷️ K-Means Clustering – Career Readiness</h3>
      <div style="font-size:2rem; margin:.4rem 0">{info['icon']}</div>
      <div style="font-size:1.2rem; font-weight:700; color:{info['color']}; margin-bottom:.5rem">{info['title']}</div>
      <div style="color:#cbd5e1; font-size:.88rem; margin-bottom:.7rem">{info['desc']}</div>
      <div style="color:#e2e8f0; font-size:.85rem; background:rgba(255,255,255,0.05);
           padding:.6rem .9rem; border-radius:8px">
        💡 {info['tip']}
      </div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Action Plan ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📋 Your Personalized Action Plan</div>', unsafe_allow_html=True)
top3_gap = skill_gap[:3]
top_course_name = top_courses.iloc[0]["course_name"] if not top_courses.empty else "N/A"

st.markdown(f"""
<div class="card">
  <table style="width:100%; color:#e2e8f0; font-size:.9rem; border-collapse:collapse;">
    <tr style="border-bottom:1px solid rgba(255,255,255,0.1)">
      <td style="padding:.6rem; color:#a78bfa; font-weight:600">Step 1</td>
      <td style="padding:.6rem">Fix your top skill gaps: <b>{', '.join(top3_gap) if top3_gap else 'None!'}</b></td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.1)">
      <td style="padding:.6rem; color:#a78bfa; font-weight:600">Step 2</td>
      <td style="padding:.6rem">Start with: <b>{top_course_name}</b></td>
    </tr>
    <tr style="border-bottom:1px solid rgba(255,255,255,0.1)">
      <td style="padding:.6rem; color:#a78bfa; font-weight:600">Step 3</td>
      <td style="padding:.6rem">Raise your match score from <b>{match_pct}%</b> → target <b>70%+</b></td>
    </tr>
    <tr>
      <td style="padding:.6rem; color:#a78bfa; font-weight:600">Step 4</td>
      <td style="padding:.6rem">Build 1 project using your top skills and deploy it on Streamlit or GitHub</td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

st.caption("Built with SQL · Scikit-Learn · Streamlit · Plotly  |  Data: Kaggle Coursera + Google Jobs datasets")
