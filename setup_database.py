"""
Phase 1 - Database Setup Script
Creates a SQLite database and loads the 3 CSV files into it.
Tables created:
  - students
  - job_postings
  - courses
"""

import sqlite3
import csv
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DB_PATH      = os.path.join(BASE_DIR, "database", "skill_gap.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# CREATE TABLES
# ─────────────────────────────────────────────────────────────────────────────
CREATE_STUDENTS = """
CREATE TABLE IF NOT EXISTS students (
    student_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    college         TEXT,
    branch          TEXT,
    year            INTEGER,
    cgpa            REAL,
    target_role     TEXT,
    current_skills  TEXT   -- pipe-separated skill list
);
"""

CREATE_JOBS = """
CREATE TABLE IF NOT EXISTS job_postings (
    job_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_title       TEXT NOT NULL,
    company         TEXT NOT NULL,
    location        TEXT,
    required_skills TEXT,  -- pipe-separated skill list
    experience_level TEXT
);
"""

CREATE_COURSES = """
CREATE TABLE IF NOT EXISTS courses (
    course_id       TEXT PRIMARY KEY,
    course_name     TEXT NOT NULL,
    platform        TEXT,
    instructor      TEXT,
    skills_taught   TEXT,  -- pipe-separated skill list
    rating          REAL,
    duration_hours  INTEGER,
    difficulty      TEXT,
    url             TEXT
);
"""

CREATE_SKILL_WEIGHTS = """
CREATE TABLE IF NOT EXISTS skill_weights (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL,
    skill       TEXT NOT NULL,
    frequency   REAL,          -- % of job postings that mention this skill
    tier        TEXT           -- 'Must Learn' | 'Should Learn' | 'Can Learn Later'
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CSV → TABLE
# ─────────────────────────────────────────────────────────────────────────────
def load_csv_to_table(conn, csv_filename, table_name, columns):
    filepath = os.path.join(DATA_DIR, csv_filename)
    cursor = conn.cursor()

    # Clear existing rows so re-runs stay idempotent
    cursor.execute(f"DELETE FROM {table_name};")

    placeholders = ", ".join(["?" for _ in columns])
    col_list     = ", ".join(columns)
    sql          = f"INSERT INTO {table_name} ({col_list}) VALUES ({placeholders})"

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = [tuple(row[c] for c in columns) for row in reader]

    cursor.executemany(sql, rows)
    conn.commit()
    print(f"  ✅  Loaded {len(rows)} rows into [{table_name}]")


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY: Run a few SQL queries and print results
# ─────────────────────────────────────────────────────────────────────────────
def run_verification_queries(conn):
    cursor = conn.cursor()

    print("\n" + "─" * 55)
    print("📊  VERIFICATION QUERIES")
    print("─" * 55)

    # 1. Row counts
    for table in ["students", "job_postings", "courses"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"  📋  {table:20s} → {count} rows")

    # 2. Sample student
    print("\n  👤  Sample Student:")
    cursor.execute("SELECT student_id, name, target_role, current_skills FROM students LIMIT 1;")
    row = cursor.fetchone()
    print(f"      ID: {row[0]} | Name: {row[1]} | Target: {row[2]}")
    print(f"      Skills: {row[3]}")

    # 3. Jobs per role
    print("\n  💼  Internship Openings per Role:")
    cursor.execute("""
        SELECT job_title, COUNT(*) as openings
        FROM job_postings
        GROUP BY job_title
        ORDER BY openings DESC;
    """)
    for r in cursor.fetchall():
        print(f"      {r[0]:30s} {r[1]} opening(s)")

    # 4. Courses per platform
    print("\n  🎓  Courses per Platform:")
    cursor.execute("""
        SELECT platform, COUNT(*) as total
        FROM courses
        GROUP BY platform
        ORDER BY total DESC;
    """)
    for r in cursor.fetchall():
        print(f"      {r[0]:20s} {r[1]} course(s)")

    # 5. Top rated courses
    print("\n  ⭐  Top 5 Highest Rated Courses:")
    cursor.execute("""
        SELECT course_name, platform, rating
        FROM courses
        ORDER BY rating DESC
        LIMIT 5;
    """)
    for r in cursor.fetchall():
        print(f"      [{r[2]}] {r[0]} ({r[1]})")

    # 6. Students per target role
    print("\n  🎯  Students per Target Role:")
    cursor.execute("""
        SELECT target_role, COUNT(*) as count
        FROM students
        GROUP BY target_role
        ORDER BY count DESC;
    """)
    for r in cursor.fetchall():
        print(f"      {r[0]:30s} {r[1]} student(s)")

    print("─" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n🗄️   Setting up SQLite database...\n")
    print(f"  📁  DB Location: {DB_PATH}\n")

    conn = get_connection()

    # Create tables
    cursor = conn.cursor()
    cursor.execute(CREATE_STUDENTS)
    cursor.execute(CREATE_JOBS)
    cursor.execute(CREATE_COURSES)
    cursor.execute(CREATE_SKILL_WEIGHTS)
    conn.commit()
    print("  ✅  Tables created (students, job_postings, courses, skill_weights)\n")

    # Load data
    print("  📥  Loading CSV data into database...\n")
    load_csv_to_table(conn, "students.csv",      "students",      ["student_id", "name", "college", "branch", "year", "cgpa", "target_role", "current_skills"])
    load_csv_to_table(conn, "job_postings.csv",  "job_postings",  ["job_title", "company", "location", "required_skills", "experience_level"])
    load_csv_to_table(conn, "courses.csv",        "courses",       ["course_id", "course_name", "platform", "instructor", "skills_taught", "rating", "duration_hours", "difficulty", "url"])
    load_csv_to_table(conn, "skill_weights.csv",  "skill_weights", ["role", "skill", "frequency", "tier"])

    # Verify
    run_verification_queries(conn)

    conn.close()
    print("✅  Phase 1 Complete! Database is ready.\n")


if __name__ == "__main__":
    main()
