"""
generate_data.py
Generates synthetic student profiles (students.csv).
Job postings and courses come from real Kaggle datasets —
see data/prepare_kaggle_data.py for those.
"""

import csv
import random
import os

# ─────────────────────────────────────────────────────────────────────────────
# MASTER SKILL BANK  (used to sprinkle extra random skills onto students)
# ─────────────────────────────────────────────────────────────────────────────
ALL_SKILLS = [
    "Python", "SQL", "Excel", "PowerBI", "Tableau", "R",
    "Machine Learning", "Statistics", "Data Visualization", "EDA",
    "Pandas", "NumPy", "Scikit-Learn", "Matplotlib", "Seaborn",
    "Product Roadmap", "A/B Testing", "User Research", "Figma",
    "Agile", "Scrum", "JIRA", "Market Research", "Go-To-Market",
    "Business Analysis", "Requirements Gathering", "Stakeholder Management",
    "Financial Modeling", "Consulting Frameworks", "Deck Building",
    "PowerPoint", "Communication", "Strategic Thinking",
    "MySQL", "PostgreSQL", "Git", "APIs", "ETL",
    "Google Analytics", "CRM", "Hypothesis Testing",
    "Regression Analysis", "Clustering", "Time Series Analysis", "Forecasting",
]

# ─────────────────────────────────────────────────────────────────────────────
# STUDENT PROFILE DATA
# ─────────────────────────────────────────────────────────────────────────────
STUDENT_NAMES = [
    "Aarav Sharma", "Priya Nair", "Rahul Gupta", "Sneha Iyer", "Arjun Mehta",
    "Kavya Reddy", "Rohan Verma", "Ananya Singh", "Vikram Patel", "Ishaan Kapoor",
    "Pooja Menon", "Aditya Joshi", "Neha Agarwal", "Karan Malhotra", "Riya Desai",
    "Siddharth Bose", "Meera Pillai", "Akash Trivedi", "Shreya Ghosh", "Varun Tiwari",
    "Divya Pandey", "Nikhil Saxena", "Swati Kulkarni", "Manish Rana", "Tanvi Choudhary",
    "Harsh Srivastava", "Apurva Jain", "Rajat Chauhan", "Deepika Nambiar", "Ayaan Mirza",
    "Simran Kaur", "Dhruv Bhatt", "Prachi Goyal", "Yash Bansal", "Kritika Thakur",
    "Abhinav Dubey", "Richa Bhat", "Mayank Mishra", "Nidhi Shah", "Tushar Rawat",
    "Garima Chaturvedi", "Kunal Patil", "Pallavi Rajan", "Sourav Das", "Ankita Mohan",
    "Aakash Yadav", "Ruchika Sethi", "Parth Dixit", "Vandana Gupta", "Samar Khan",
]

COLLEGES = [
    "IIT Delhi", "IIT Bombay", "IIT Madras", "NIT Trichy", "BITS Pilani",
    "NSUT Delhi", "DTU Delhi", "VIT Vellore", "Manipal Institute of Technology",
    "IIIT Hyderabad", "Jadavpur University", "Anna University",
]

BRANCHES = [
    "Computer Science", "Electrical Engineering", "Electronics & Communication",
    "Mechanical Engineering", "Civil Engineering", "Information Technology",
    "Chemical Engineering",
]

TARGET_ROLES = [
    "Data Analyst", "Data Scientist", "Product Manager",
    "Management Consultant", "Business Analyst",
]

# Base skills each student knows depending on their target role
SKILL_POOLS = {
    "Data Analyst":          ["Python", "SQL", "Excel", "EDA", "Statistics", "Pandas", "NumPy"],
    "Data Scientist":        ["Python", "Machine Learning", "SQL", "Pandas", "NumPy", "Statistics", "Git"],
    "Product Manager":       ["Communication", "Excel", "PowerPoint", "Market Research", "Agile", "Strategic Thinking"],
    "Management Consultant": ["Excel", "PowerPoint", "Communication", "Strategic Thinking", "Financial Modeling"],
    "Business Analyst":      ["Excel", "SQL", "PowerPoint", "Communication", "Business Analysis"],
}

random.seed(42)


def generate_students(n=50):
    students = []
    for i in range(1, n + 1):
        name        = random.choice(STUDENT_NAMES)
        target_role = random.choice(TARGET_ROLES)
        base_skills = SKILL_POOLS[target_role]

        # Give each student 2–5 skills relevant to their target role
        known  = random.sample(base_skills, k=random.randint(2, min(5, len(base_skills))))
        # Sprinkle 0–2 random extra skills
        extras = random.sample(ALL_SKILLS, k=random.randint(0, 2))

        current_skills = list(set(known + extras))
        students.append({
            "student_id":     f"S{i:03d}",
            "name":           name,
            "college":        random.choice(COLLEGES),
            "branch":         random.choice(BRANCHES),
            "year":           random.choice([2, 3]),
            "cgpa":           round(random.uniform(6.5, 9.5), 2),
            "target_role":    target_role,
            "current_skills": "|".join(current_skills),
        })
    return students


# ─────────────────────────────────────────────────────────────────────────────
# WRITE students.csv  (run this file directly to regenerate)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(DATA_DIR, "students.csv")

    students = generate_students(50)
    fieldnames = ["student_id", "name", "college", "branch", "year", "cgpa", "target_role", "current_skills"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(students)

    print(f"✅  students.csv written → {len(students)} rows  ({out_path})")
