"""
analyzer.py  –  Phase 2: The Brain of the Skill-Gap Analyzer
=============================================================
Three ML layers:
  Layer 1 – Content-Based Course Recommender  (TF-IDF + Cosine Similarity)
  Layer 2 – Employability Score Predictor     (Logistic Regression)
  Layer 3 – Student Segmentation              (K-Means Clustering)

All data is read from skill_gap.db via SQL queries.
"""

import sqlite3
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CONNECTION
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "database", "skill_gap.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn


# ═════════════════════════════════════════════════════════════════════════════
# SQL HELPERS  –  Pure SQL queries, no hardcoding
# ═════════════════════════════════════════════════════════════════════════════

def fetch_student(conn, student_id: str) -> dict:
    """Return a single student's profile from the DB."""
    cur = conn.execute(
        "SELECT * FROM students WHERE student_id = ?;", (student_id,)
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Student '{student_id}' not found in database.")
    return dict(row)


def fetch_all_students(conn) -> pd.DataFrame:
    """Return all students as a DataFrame (used for Layer 2 & 3)."""
    return pd.read_sql_query("SELECT * FROM students;", conn)


def fetch_required_skills_for_role(conn, role: str) -> list[str]:
    """
    SQL QUERY 1:
    For a given target role, collect ALL required_skills from every
    matching job posting, flatten and count them.
    Returns a deduplicated, frequency-sorted list of required skills.
    """
    cur = conn.execute(
        """
        SELECT required_skills
        FROM   job_postings
        WHERE  LOWER(job_title) = LOWER(?);
        """,
        (role,)
    )
    rows = cur.fetchall()

    skill_freq: dict[str, int] = {}
    for row in rows:
        for skill in row["required_skills"].split("|"):
            s = skill.strip().title()
            if s:
                skill_freq[s] = skill_freq.get(s, 0) + 1

    # Sort by frequency (most demanded skill first)
    sorted_skills = sorted(skill_freq, key=skill_freq.get, reverse=True)
    return sorted_skills


def fetch_all_courses(conn) -> pd.DataFrame:
    """
    SQL QUERY 2:
    Return all courses with their skills, rating, difficulty, and URL.
    Also pre-computes a cleaned text column for TF-IDF.
    """
    df = pd.read_sql_query(
        """
        SELECT course_id, course_name, platform, instructor,
               skills_taught, rating, duration_hours, difficulty, url
        FROM   courses
        WHERE  skills_taught IS NOT NULL
          AND  skills_taught != '';
        """,
        conn
    )
    # Create a single text blob of all skills (pipe → space, for TF-IDF)
    df["skills_text"] = df["skills_taught"].str.replace("|", " ", regex=False)
    return df


def fetch_top_skills_summary(conn) -> pd.DataFrame:
    """
    SQL QUERY 3  (Bonus analytical query):
    For every role, count how many job postings mention each skill.
    Returns a DataFrame with role / skill / frequency.
    """
    cur = conn.execute("SELECT job_title, required_skills FROM job_postings;")
    rows = cur.fetchall()

    records = []
    for row in rows:
        for skill in row["required_skills"].split("|"):
            s = skill.strip().title()
            if s:
                records.append({"role": row["job_title"], "skill": s})

    df = pd.DataFrame(records)
    return (
        df.groupby(["role", "skill"])
          .size()
          .reset_index(name="frequency")
          .sort_values(["role", "frequency"], ascending=[True, False])
    )


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 1 – COURSE RECOMMENDER  (TF-IDF + Cosine Similarity)
# ═════════════════════════════════════════════════════════════════════════════

def recommend_courses(skill_gap: list[str], courses_df: pd.DataFrame,
                      top_n: int = 3) -> pd.DataFrame:
    """
    Convert the skill_gap list and all course skill texts into TF-IDF vectors.
    Use Cosine Similarity to rank courses by how well they cover missing skills.
    Returns the top_n best-matching courses.
    """
    if not skill_gap:
        return pd.DataFrame()   # no gap → nothing to recommend

    # The "query" is the student's missing skills joined as a sentence
    query_text = " ".join(skill_gap)

    # Combine query + all course texts into one corpus for consistent vectorization
    all_texts = [query_text] + courses_df["skills_text"].tolist()

    vectorizer  = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # query vector  = first row; course vectors = remaining rows
    query_vec   = tfidf_matrix[0:1]
    course_vecs = tfidf_matrix[1:]

    scores = cosine_similarity(query_vec, course_vecs).flatten()

    courses_df = courses_df.copy()
    courses_df["match_score"] = scores

    top = (
        courses_df.sort_values("match_score", ascending=False)
                  .head(top_n)
                  .reset_index(drop=True)
    )
    return top[["course_name", "platform", "difficulty", "rating",
                "duration_hours", "match_score", "url"]]


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2 – EMPLOYABILITY SCORE  (Logistic Regression)
# ═════════════════════════════════════════════════════════════════════════════

def build_employability_model(students_df: pd.DataFrame,
                              conn) -> LogisticRegression:
    """
    Train a Logistic Regression model on ALL students.

    Features:
      - skill_match_score  : % of required job skills the student already has
      - cgpa               : student's CGPA
      - skills_known_count : total number of skills the student has listed

    Label strategy:
      - We use the MEDIAN skill_match_score as the split threshold.
        Students above the median are labelled "employable=1" (top half),
        students below are labelled "employable=0" (bottom half).
        This guarantees both classes always exist regardless of the data.

    Returns the fitted LogisticRegression model + the training DataFrame.
    """
    records = []
    for _, row in students_df.iterrows():
        known    = set(s.strip().title() for s in row["current_skills"].split("|") if s.strip())
        required = set(fetch_required_skills_for_role(conn, row["target_role"])[:20])

        if not required:
            continue

        match_score = len(known & required) / len(required) * 100
        records.append({
            "skill_match_score":  round(match_score, 2),
            "cgpa":               float(row["cgpa"]),
            "skills_known_count": len(known),
        })

    train_df = pd.DataFrame(records)

    # Use median as threshold so both classes are always present
    median_score = train_df["skill_match_score"].median()
    train_df["employable"] = (train_df["skill_match_score"] >= median_score).astype(int)

    X = train_df[["skill_match_score", "cgpa", "skills_known_count"]].values
    y = train_df["employable"].values

    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    return model, train_df


def predict_employability(model: LogisticRegression,
                          skill_match_score: float,
                          cgpa: float,
                          skills_known_count: int) -> dict:
    """
    Use the trained Logistic Regression model to predict:
      - employable (0/1)
      - probability of being employable (0–100%)
    """
    X = np.array([[skill_match_score, cgpa, skills_known_count]])
    prob       = model.predict_proba(X)[0][1]   # probability of class=1
    prediction = int(model.predict(X)[0])
    return {
        "employable":   prediction,
        "probability":  round(prob * 100, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 3 – STUDENT SEGMENTATION  (K-Means Clustering)
# ═════════════════════════════════════════════════════════════════════════════

CLUSTER_LABELS = {
    0: "High Achiever 🏆",
    1: "On Track 📈",
    2: "Needs Focus 🎯",
}


def build_clustering_model(students_df: pd.DataFrame, conn,
                           n_clusters: int = 3) -> tuple:
    """
    Cluster all students using K-Means on 3 features:
      - skill_match_score
      - cgpa
      - skills_known_count

    Returns (fitted KMeans model, feature DataFrame with cluster labels).
    """
    records = []
    for _, row in students_df.iterrows():
        known    = set(s.strip().title() for s in row["current_skills"].split("|") if s.strip())
        required = set(fetch_required_skills_for_role(conn, row["target_role"])[:20])
        match    = len(known & required) / len(required) * 100 if required else 0

        records.append({
            "student_id":         row["student_id"],
            "skill_match_score":  round(match, 2),
            "cgpa":               float(row["cgpa"]),
            "skills_known_count": len(known),
        })

    feat_df = pd.DataFrame(records)
    X       = feat_df[["skill_match_score", "cgpa", "skills_known_count"]].values

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    feat_df["cluster"]       = km.fit_predict(X)
    feat_df["cluster_label"] = feat_df["cluster"].map(CLUSTER_LABELS)

    return km, feat_df


def get_student_cluster(km_model: KMeans, cluster_df: pd.DataFrame,
                        student_id: str) -> dict:
    """Return the cluster info for a specific student."""
    row = cluster_df[cluster_df["student_id"] == student_id]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "cluster_id":    int(r["cluster"]),
        "cluster_label": r["cluster_label"],
        "peers_in_cluster": int((cluster_df["cluster"] == r["cluster"]).sum()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN  –  Run analysis for a sample student and print full results
# ═════════════════════════════════════════════════════════════════════════════

def analyze_student(student_id: str = "S001"):
    conn        = get_conn()
    students_df = fetch_all_students(conn)

    # ── Fetch student profile ──────────────────────────────────────────────
    student = fetch_student(conn, student_id)
    student_skills  = [s.strip().title() for s in student["current_skills"].split("|") if s.strip()]
    target_role     = student["target_role"]

    print("\n" + "═" * 60)
    print(f"  SKILL GAP ANALYSIS  –  {student['name']}  ({student_id})")
    print("═" * 60)
    print(f"  🎓  College      : {student['college']}")
    print(f"  📚  Branch       : {student['branch']}  |  Year {student['year']}")
    print(f"  📊  CGPA         : {student['cgpa']}")
    print(f"  🎯  Target Role  : {target_role}")
    print(f"  ✅  Known Skills : {', '.join(student_skills)}")

    # ── SQL Query: Required skills for role ───────────────────────────────
    required_skills = fetch_required_skills_for_role(conn, target_role)[:20]
    student_set     = set(student_skills)
    required_set    = set(required_skills)
    known_set       = student_set & required_set
    skill_gap       = list(required_set - student_set)

    skill_match_pct = round(len(known_set) / len(required_set) * 100, 1) if required_set else 0

    print(f"\n  💼  Top Required Skills ({target_role}):")
    print(f"      {', '.join(required_skills[:10])}")
    print(f"\n  ✅  Skills You Already Have  ({len(known_set)}):")
    print(f"      {', '.join(sorted(known_set)) if known_set else 'None matched'}")
    print(f"\n  ❌  Your Skill Gap  ({len(skill_gap)} missing skills):")
    print(f"      {', '.join(sorted(skill_gap)) if skill_gap else 'No gap! You are fully ready.'}")
    print(f"\n  📊  Skill Match Score : {skill_match_pct}%")

    # ── Layer 1: Course Recommendations ───────────────────────────────────
    print("\n" + "─" * 60)
    print("  LAYER 1 – COURSE RECOMMENDATIONS  (TF-IDF + Cosine Similarity)")
    print("─" * 60)
    courses_df   = fetch_all_courses(conn)
    top_courses  = recommend_courses(skill_gap, courses_df, top_n=3)

    if top_courses.empty:
        print("  🎉  No courses needed — you already have all required skills!")
    else:
        for i, row in top_courses.iterrows():
            print(f"\n  #{i+1}  {row['course_name']}")
            print(f"       Platform   : {row['platform']}  |  {row['difficulty']}")
            print(f"       Rating     : ⭐ {row['rating']}  |  {row['duration_hours']}h")
            print(f"       Match Score: {round(row['match_score'] * 100, 1)}%")
            print(f"       🔗  {row['url']}")

    # ── Layer 2: Employability Score ───────────────────────────────────────
    print("\n" + "─" * 60)
    print("  LAYER 2 – EMPLOYABILITY PREDICTOR  (Logistic Regression)")
    print("─" * 60)
    lr_model, _ = build_employability_model(students_df, conn)
    emp_result  = predict_employability(
        lr_model,
        skill_match_score  = skill_match_pct,
        cgpa               = float(student["cgpa"]),
        skills_known_count = len(student_skills)
    )
    status = "✅  LIKELY TO PASS RESUME SCREENING" if emp_result["employable"] else "⚠️  AT RISK — Skill Gap too large"
    print(f"\n  📈  Employability Probability : {emp_result['probability']}%")
    print(f"  {status}")
    if emp_result["probability"] < 70:
        print(f"  💡  Close your skill gap to increase this score significantly.")

    # ── Layer 3: Cluster Segmentation ──────────────────────────────────────
    print("\n" + "─" * 60)
    print("  LAYER 3 – STUDENT SEGMENTATION  (K-Means Clustering)")
    print("─" * 60)
    km_model, cluster_df = build_clustering_model(students_df, conn)
    cluster_info         = get_student_cluster(km_model, cluster_df, student_id)
    print(f"\n  🏷️   Your Segment  : {cluster_info.get('cluster_label', 'N/A')}")
    print(f"  👥  Peers in your cluster : {cluster_info.get('peers_in_cluster', 'N/A')} students")

    # Show cluster distribution
    print("\n  📊  Overall Student Distribution:")
    dist = cluster_df.groupby("cluster_label").agg(
        count      = ("student_id",        "count"),
        avg_match  = ("skill_match_score",  "mean"),
        avg_cgpa   = ("cgpa",               "mean"),
    ).reset_index()
    for _, dr in dist.iterrows():
        print(f"       {dr['cluster_label']:22s}  "
              f"Students: {int(dr['count']):3d}  |  "
              f"Avg Match: {dr['avg_match']:5.1f}%  |  "
              f"Avg CGPA: {dr['avg_cgpa']:.2f}")

    # ── Actionable Summary ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  📋  ACTION PLAN SUMMARY")
    print("═" * 60)
    print(f"  1.  Fix your top 3 skill gaps : {', '.join(sorted(skill_gap)[:3])}")
    if not top_courses.empty:
        print(f"  2.  Start with this course    : {top_courses.iloc[0]['course_name']}")
    print(f"  3.  Your current match score  : {skill_match_pct}% → Target: 70%+")
    print(f"  4.  Employability probability : {emp_result['probability']}%")
    print("═" * 60 + "\n")

    conn.close()
    return {
        "student":          student,
        "skill_gap":        skill_gap,
        "skill_match_pct":  skill_match_pct,
        "top_courses":      top_courses,
        "employability":    emp_result,
        "cluster":          cluster_info,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run for a sample student — change "S001" to any ID (S001 to S050)
    analyze_student("S001")

    print("\n--- Running for a second student for comparison ---")
    analyze_student("S010")
