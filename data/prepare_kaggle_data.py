"""
Phase 1 – Kaggle Data Preparation Script (v2)
Reads the raw Kaggle datasets, cleans them, builds weighted skill profiles,
and saves formatted CSVs into the /data folder.

Input files (in /PROJECT root):
  - coursera_course_dataset_v2_no_null.csv  → Coursera courses
  - archive/postings.csv                    → LinkedIn job postings (3.3M rows)
  - udemy_courses.csv                       → Udemy courses

Output files (in /data):
  - courses.csv          (Coursera + Udemy combined)
  - job_postings.csv     (filtered LinkedIn postings with weighted skills)
  - skill_weights.csv    (per-role skill frequency weights for UI tiers)
  - students.csv         (synthetic — no Kaggle source)
"""

import pandas as pd
import re
import random
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
RAW_DIR  = "/Users/abdullahkhan/Desktop/PROJECT"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

COURSERA_RAW  = os.path.join(RAW_DIR, "coursera_course_dataset_v2_no_null.csv")
LINKEDIN_RAW  = os.path.join(RAW_DIR, "archive", "postings.csv")
UDEMY_RAW     = os.path.join(RAW_DIR, "udemy_courses.csv")

# ─────────────────────────────────────────────────────────────────────────────
# TARGET ROLES  (expanded to cover more friends' career paths)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_ROLE_KEYWORDS = {
    "Data Analyst":          ["data analyst", "analytics analyst", "bi analyst", "business intelligence analyst"],
    "Data Scientist":        ["data scientist", "machine learning engineer", "ml engineer", "ai engineer"],
    "Software Engineer":     ["software engineer", "software developer", "backend developer", "backend engineer",
                              "sde", "software development engineer"],
    "Web Developer":         ["web developer", "frontend developer", "front-end developer", "full stack developer",
                              "full-stack developer", "fullstack developer", "react developer", "javascript developer"],
    "Android/iOS Developer": ["android developer", "ios developer", "mobile developer", "flutter developer",
                              "react native developer"],
    "Business Analyst":      ["business analyst", "business systems analyst"],
    "Product Manager":       ["product manager", "product management", "associate product manager", "apm"],
    "Management Consultant": ["management consultant", "strategy analyst", "strategy consultant", "consultant"],
    "Marketing Analyst":     ["marketing analyst", "digital marketing", "seo specialist", "growth analyst",
                              "marketing associate", "content strategist"],
    "HR Analyst":            ["hr analyst", "human resources analyst", "talent acquisition", "people analyst",
                              "hr generalist", "recruiter"],
    "UX/UI Designer":        ["ux designer", "ui designer", "ux/ui", "ui/ux", "product designer",
                              "user experience designer"],
    "Financial Analyst":     ["financial analyst", "finance analyst", "investment analyst", "equity analyst",
                              "accounting analyst", "fp&a analyst"],
}

# ─────────────────────────────────────────────────────────────────────────────
# SKILL KEYWORD EXTRACTION  (for LinkedIn postings that use free-text skills_desc)
# Maps keyword found in text → canonical skill name
# ─────────────────────────────────────────────────────────────────────────────
SKILL_KEYWORDS = {
    # Programming
    "python":           "Python",
    "r programming":    "R",
    r"\br\b":           "R",
    r"java\b":          "Java",
    "javascript":       "JavaScript",
    "typescript":       "TypeScript",
    r"c\+\+":           "C++",
    r"\bc#\b":          "C#",
    "swift":            "Swift",
    "kotlin":           "Kotlin",
    "flutter":          "Flutter",
    "react native":     "React Native",
    "golang":           "Go",

    # Web/Frontend
    "react":            "React",
    "angular":          "Angular",
    "vue":              "Vue.js",
    r"node\.?js":       "Node.js",
    "html":             "HTML",
    "css":              "CSS",
    "django":           "Django",
    "flask":            "Flask",
    "rest api":         "REST APIs",
    "graphql":          "GraphQL",

    # Data / Analytics
    "sql":              "SQL",
    "mysql":            "MySQL",
    "postgresql":       "PostgreSQL",
    "mongodb":          "MongoDB",
    "nosql":            "NoSQL",
    "excel":            "Excel",
    "power bi":         "Power BI",
    "powerbi":          "Power BI",
    "tableau":          "Tableau",
    "pandas":           "Pandas",
    "numpy":            "NumPy",
    "matplotlib":       "Matplotlib",
    "seaborn":          "Seaborn",
    "data visualization": "Data Visualization",
    "data analysis":    "Data Analysis",
    "statistics":       "Statistics",
    "statistical":      "Statistics",
    "machine learning": "Machine Learning",
    "deep learning":    "Deep Learning",
    "tensorflow":       "TensorFlow",
    "pytorch":          "PyTorch",
    "scikit":           "Scikit-Learn",
    "nlp":              "NLP",
    "natural language": "NLP",
    "computer vision":  "Computer Vision",
    "time series":      "Time Series",
    "a/b testing":      "A/B Testing",
    "a/b test":         "A/B Testing",

    # Cloud / DevOps
    "aws":              "AWS",
    "azure":            "Azure",
    "google cloud":     "GCP",
    "gcp":              "GCP",
    "docker":           "Docker",
    "git\b":            "Git",
    "github":           "Git",
    "linux":            "Linux",

    # Business / PM
    "agile":            "Agile",
    "scrum":            "Scrum",
    "jira":             "JIRA",
    "product management": "Product Management",
    "product roadmap":  "Product Roadmap",
    "user research":    "User Research",
    "market research":  "Market Research",
    "stakeholder":      "Stakeholder Management",
    "communication":    "Communication",
    "presentation":     "Presentation",
    "powerpoint":       "PowerPoint",
    "excel":            "Excel",
    "financial model":  "Financial Modeling",
    "financial analysis": "Financial Analysis",
    "forecasting":      "Forecasting",
    "business analysis": "Business Analysis",
    "strategy":         "Strategic Thinking",
    "consulting":       "Consulting Frameworks",
    "crm":              "CRM",

    # Marketing
    "seo":              "SEO",
    "google analytics": "Google Analytics",
    "social media":     "Social Media",
    "content":          "Content Marketing",
    "digital marketing": "Digital Marketing",
    "email marketing":  "Email Marketing",
    "paid ads":         "Paid Advertising",

    # HR
    "talent acquisition": "Talent Acquisition",
    "recruitment":      "Recruitment",
    "onboarding":       "Onboarding",
    "hris":             "HRIS",
    "performance management": "Performance Management",
    "employee relations": "Employee Relations",

    # UX/UI
    "figma":            "Figma",
    "sketch":           "Sketch",
    "adobe xd":         "Adobe XD",
    "wireframe":        "Wireframing",
    "prototyp":         "Prototyping",
    "user testing":     "User Testing",
    "usability":        "Usability Testing",
    "design system":    "Design Systems",
    "accessibility":    "Accessibility",

    # Finance
    "financial statement": "Financial Statements",
    "accounting":       "Accounting",
    "valuation":        "Valuation",
    "bloomberg":        "Bloomberg",
    "investment":       "Investment Analysis",
    "equity":           "Equity Research",
    "risk management":  "Risk Management",
}

# Senior-level title keywords — filter these OUT
SENIOR_TITLE_KEYWORDS = [
    "senior", "sr.", " sr ", "lead ", "principal", "director",
    "head of", " vp ", "vice president", "chief", " ii ", " iii ",
    "manager,", "manager -",  # keep "product manager" but filter "engineering manager"
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: normalize a skill key for deduplication
# ─────────────────────────────────────────────────────────────────────────────
def normalize_key(skill: str) -> str:
    """Strip separators and lowercase for dedup-safe comparison."""
    return re.sub(r'[\s_\-]+', '', skill.strip().lower())


def extract_skills_from_text(text: str) -> list[str]:
    """
    Scan free-text job description for known skill keywords.
    Returns a deduplicated list of canonical skill names.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    text_lower = text.lower()
    found = {}
    for pattern, canonical in SKILL_KEYWORDS.items():
        if re.search(pattern, text_lower):
            key = normalize_key(canonical)
            if key not in found:
                found[key] = canonical
    return list(found.values())


def map_to_target_role(title: str):
    t = title.lower()
    # Exclude senior roles globally
    if any(kw in t for kw in SENIOR_TITLE_KEYWORDS):
        return None
    for role, keywords in TARGET_ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return role
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. PREPARE COURSES.CSV  (Coursera + Udemy combined)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_courses():
    print("📚  Processing Coursera dataset...")
    df_c = pd.read_csv(COURSERA_RAW)

    def extract_difficulty(meta):
        if pd.isna(meta):       return "Intermediate"
        if "Beginner" in meta:  return "Beginner"
        if "Advanced" in meta:  return "Advanced"
        return "Intermediate"

    def extract_duration(meta):
        if pd.isna(meta): return 20
        hours_match  = re.search(r'(\d+)\s*-?\s*(\d+)?\s*Hour',  str(meta), re.I)
        months_match = re.search(r'(\d+)\s*-?\s*(\d+)?\s*Month', str(meta), re.I)
        if hours_match:
            lo, hi = int(hours_match.group(1)), int(hours_match.group(2) or hours_match.group(1))
            return (lo + hi) // 2
        if months_match:
            lo, hi = int(months_match.group(1)), int(months_match.group(2) or months_match.group(1))
            return ((lo + hi) // 2) * 10
        return 20

    def clean_skills(s):
        if pd.isna(s): return ""
        return "|".join(p.strip() for p in str(s).split(",") if p.strip())

    coursera = pd.DataFrame({
        "course_name":    df_c["Title"].str.strip(),
        "platform":       "Coursera",
        "instructor":     df_c["Organization"].str.strip(),
        "skills_taught":  df_c["Skills"].apply(clean_skills),
        "rating":         pd.to_numeric(df_c["Ratings"], errors="coerce").fillna(4.0).round(1),
        "duration_hours": df_c["Metadata"].apply(extract_duration),
        "difficulty":     df_c["Metadata"].apply(extract_difficulty),
        "url":            "https://www.coursera.org/search?query=" + df_c["Title"].str.strip().str.replace(" ", "+"),
    })
    coursera = coursera[coursera["skills_taught"] != ""]
    print(f"  ✅  Coursera: {len(coursera)} courses")

    # ── Udemy ─────────────────────────────────────────────────────────────────
    print("📚  Processing Udemy dataset...")
    df_u = pd.read_csv(UDEMY_RAW, on_bad_lines="skip", low_memory=False)

    def udemy_level(lvl):
        if pd.isna(lvl): return "Intermediate"
        l = str(lvl).lower()
        if "beginner" in l: return "Beginner"
        if "expert"   in l: return "Advanced"
        return "Intermediate"

    def udemy_skills(row):
        """Extract skills from headline + objectives text."""
        text = " ".join([
            str(row.get("headline", "")),
            str(row.get("objectives", "")),
            str(row.get("curriculum",  "")),
        ])
        skills = extract_skills_from_text(text)
        return "|".join(skills) if skills else ""

    # Only use paid courses with rating ≥ 4.0 and decent subscriber base
    df_u["rating"] = pd.to_numeric(df_u.get("rating", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df_u["num_subscribers"] = pd.to_numeric(df_u.get("num_subscribers", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df_u = df_u[(df_u["rating"] >= 4.0) & (df_u["num_subscribers"] >= 500)].copy()

    df_u["skills_taught"] = df_u.apply(udemy_skills, axis=1)
    df_u = df_u[df_u["skills_taught"] != ""].copy()

    udemy = pd.DataFrame({
        "course_name":    df_u["title"].str.strip(),
        "platform":       "Udemy",
        "instructor":     df_u["instructor_names"].fillna("Udemy Instructor").str.strip(),
        "skills_taught":  df_u["skills_taught"],
        "rating":         df_u["rating"].round(1),
        "duration_hours": 10,   # Udemy doesn't provide this cleanly
        "difficulty":     df_u["instructional_level"].apply(udemy_level),
        "url":            df_u["url"].fillna("https://www.udemy.com"),
    })
    print(f"  ✅  Udemy: {len(udemy)} courses")

    # ── Combine ───────────────────────────────────────────────────────────────
    combined = pd.concat([coursera, udemy], ignore_index=True)
    combined["course_id"] = ["C" + str(i + 1).zfill(4) for i in range(len(combined))]
    combined = combined[["course_id", "course_name", "platform", "instructor",
                          "skills_taught", "rating", "duration_hours", "difficulty", "url"]]

    out_path = os.path.join(DATA_DIR, "courses.csv")
    combined.to_csv(out_path, index=False)
    print(f"  ✅  courses.csv → {len(combined)} rows total  (saved to {out_path})")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPARE JOB_POSTINGS.CSV  (LinkedIn postings.csv)
#    Also compute SKILL WEIGHTS per role for the priority tiers
# ─────────────────────────────────────────────────────────────────────────────
def prepare_jobs():
    print("💼  Processing LinkedIn job postings dataset (3.3M rows — this takes ~1 min)...")

    # Read only the columns we need to save memory
    usecols = ["job_id", "title", "company_name", "location",
               "formatted_experience_level", "description"]

    # Read in chunks to handle 500MB file efficiently
    CHUNK  = 100_000
    MAX_PER_ROLE = 150   # keep up to 150 postings per role for good skill coverage
    role_buckets = {role: [] for role in TARGET_ROLE_KEYWORDS}

    reader = pd.read_csv(LINKEDIN_RAW, usecols=usecols, chunksize=CHUNK,
                         low_memory=False, on_bad_lines="skip")

    total_seen = 0
    for chunk in reader:
        total_seen += len(chunk)

        # Keep Entry level + Internship + Associate (skip Director, Executive)
        if "formatted_experience_level" in chunk.columns:
            chunk = chunk[chunk["formatted_experience_level"].isin(
                ["Entry level", "Internship", "Associate", None]
            )]

        chunk = chunk.dropna(subset=["title", "description"])

        for _, row in chunk.iterrows():
            role = map_to_target_role(str(row["title"]))
            if role is None:
                continue
            if len(role_buckets[role]) >= MAX_PER_ROLE:
                continue
            skills = extract_skills_from_text(str(row["description"]))
            if len(skills) < 2:
                continue
            role_buckets[role].append({
                "job_title":        role,
                "company":          str(row.get("company_name", "")).strip(),
                "location":         str(row.get("location", "Remote")).strip(),
                "required_skills":  "|".join(skills),
                "experience_level": str(row.get("formatted_experience_level", "Entry level")),
            })

        # If all buckets are full, stop early
        if all(len(v) >= MAX_PER_ROLE for v in role_buckets.values()):
            print(f"  ⚡  All role buckets full after {total_seen:,} rows scanned — stopping early.")
            break

    # Flatten buckets
    records = []
    for rows in role_buckets.values():
        records.extend(rows)

    out = pd.DataFrame(records).reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, "job_postings.csv")
    out.to_csv(out_path, index=False)
    print(f"  ✅  job_postings.csv → {len(out)} rows  (saved to {out_path})")
    print("\n  📊  Rows per target role:")
    for role, count in out["job_title"].value_counts().items():
        print(f"       {role:30s} {count} row(s)")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE SKILL WEIGHTS PER ROLE
#    For each role, count how many postings mention each skill.
#    Normalize to a 0–100 frequency score.
#    Tiers: Must Learn (≥60), Should Learn (30–59), Can Learn Later (<30)
# ─────────────────────────────────────────────────────────────────────────────
def compute_skill_weights(job_df: pd.DataFrame):
    print("⚖️   Computing skill weights per role...")
    records = []
    for role in job_df["job_title"].unique():
        role_df = job_df[job_df["job_title"] == role]
        total   = len(role_df)
        freq    = {}
        for skills_str in role_df["required_skills"]:
            for s in str(skills_str).split("|"):
                s = s.strip()
                if s:
                    freq[s] = freq.get(s, 0) + 1

        for skill, count in freq.items():
            pct = round(count / total * 100, 1)
            if pct >= 60:
                tier = "Must Learn"
            elif pct >= 30:
                tier = "Should Learn"
            else:
                tier = "Can Learn Later"
            records.append({
                "role":       role,
                "skill":      skill,
                "frequency":  pct,
                "tier":       tier,
            })

    weights_df = pd.DataFrame(records).sort_values(["role", "frequency"], ascending=[True, False])
    out_path   = os.path.join(DATA_DIR, "skill_weights.csv")
    weights_df.to_csv(out_path, index=False)
    print(f"  ✅  skill_weights.csv → {len(weights_df)} rows  (saved to {out_path})")
    return weights_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. STUDENTS.CSV  (synthetic — for ML training data)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_students(job_df: pd.DataFrame):
    """
    Generate synthetic student profiles using the expanded role list
    so the ML models train on the same roles the app supports.
    """
    print("👤  Generating synthetic student profiles...")

    ROLES      = list(TARGET_ROLE_KEYWORDS.keys())
    COLLEGES   = ["IIT Delhi", "IIT Bombay", "IIT Madras", "NIT Trichy",
                  "BITS Pilani", "DTU Delhi", "NSUT Delhi", "VIT Vellore",
                  "Manipal Institute of Technology", "Anna University",
                  "IIIT Hyderabad", "Jadavpur University"]
    BRANCHES   = ["Computer Science", "Information Technology", "Electronics & Communication",
                  "Electrical Engineering", "Mechanical Engineering", "Chemical Engineering",
                  "Civil Engineering"]
    FIRST      = ["Aarav","Ananya","Arjun","Ishaan","Karan","Kavya","Kritika","Kunal",
                  "Meera","Pallavi","Pooja","Richa","Rohan","Samar","Simran","Sneha",
                  "Siddharth","Swati","Tanvi","Tushar","Vandana","Apurva","Abhinav",
                  "Dhruv","Garima","Ankita","Ayaan","Mayank"]
    LAST       = ["Sharma","Verma","Singh","Gupta","Kumar","Mehta","Patel","Reddy",
                  "Bhat","Joshi","Iyer","Menon","Pillai","Nair","Chaturvedi","Dubey",
                  "Kapoor","Bose","Rawat","Mirza","Khan","Malhotra","Mohan","Jain"]

    # Skill pool per role — used to generate realistic student skill sets
    ROLE_SKILLS = {
        "Data Analyst":          ["SQL","Python","Excel","Tableau","Power BI","Pandas","Statistics","Data Visualization","NumPy","Git"],
        "Data Scientist":        ["Python","Machine Learning","SQL","Statistics","Pandas","NumPy","Scikit-Learn","Deep Learning","TensorFlow","Git"],
        "Software Engineer":     ["Python","Java","C++","Data Structures","Git","SQL","REST APIs","Linux","Agile","Docker"],
        "Web Developer":         ["HTML","CSS","JavaScript","React","Node.js","Git","REST APIs","TypeScript","SQL","Vue.js"],
        "Android/iOS Developer": ["Java","Kotlin","Swift","Flutter","React Native","Git","REST APIs","Android","Agile","Firebase"],
        "Business Analyst":      ["SQL","Excel","PowerPoint","Business Analysis","Communication","Data Visualization","JIRA","Agile","Stakeholder Management","Power BI"],
        "Product Manager":       ["Product Roadmap","Agile","JIRA","Market Research","Communication","User Research","Strategic Thinking","Excel","Scrum","Presentation"],
        "Management Consultant": ["Financial Modeling","PowerPoint","Strategic Thinking","Consulting Frameworks","Excel","Communication","Data Analysis","Stakeholder Management","Presentation","Risk Management"],
        "Marketing Analyst":     ["Google Analytics","Excel","SEO","Social Media","Digital Marketing","Content Marketing","Data Analysis","PowerPoint","Email Marketing","Tableau"],
        "HR Analyst":            ["Excel","Communication","Recruitment","HRIS","Talent Acquisition","PowerPoint","Employee Relations","Onboarding","Data Analysis","Performance Management"],
        "UX/UI Designer":        ["Figma","User Research","Wireframing","Prototyping","Adobe XD","User Testing","Design Systems","HTML","CSS","Usability Testing"],
        "Financial Analyst":     ["Excel","Financial Modeling","SQL","PowerPoint","Accounting","Financial Statements","Bloomberg","Forecasting","Risk Management","Investment Analysis"],
    }

    random.seed(42)
    students = []
    for i in range(60):   # 60 students across 12 roles = 5 per role on average
        role   = ROLES[i % len(ROLES)]
        skills = ROLE_SKILLS.get(role, [])
        n      = random.randint(2, min(6, len(skills)))
        chosen = random.sample(skills, n)
        students.append({
            "student_id":     f"S{i+1:03d}",
            "name":           f"{random.choice(FIRST)} {random.choice(LAST)}",
            "college":        random.choice(COLLEGES),
            "branch":         random.choice(BRANCHES),
            "year":           random.randint(2, 4),
            "cgpa":           round(random.uniform(6.5, 9.5), 2),
            "target_role":    role,
            "current_skills": "|".join(chosen),
        })

    out = pd.DataFrame(students)
    out_path = os.path.join(DATA_DIR, "students.csv")
    out.to_csv(out_path, index=False)
    print(f"  ✅  students.csv → {len(out)} rows  (saved to {out_path})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔄  Preparing datasets for database ingestion...\n")
    prepare_courses()
    print()
    job_df = prepare_jobs()
    print()
    compute_skill_weights(job_df)
    print()
    prepare_students(job_df)
    print("\n✅  All files ready. Run setup_database.py next.\n")
