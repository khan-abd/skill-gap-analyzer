Access this web-app via https://skillgap-analyzer.streamlit.app/
=======
# SkillBridge: AI-Powered Skill Gap Analyzer 🚀

SkillBridge is an advanced, data-driven career readiness platform built to bridge the gap between student resumes and industry requirements. By leveraging Machine Learning, NLP, and massive real-world datasets, SkillBridge analyzes your profile, identifies missing skills, and recommends hyper-personalized courses to make you job-ready.

<img width="1920" height="962" alt="Screenshot 2026-05-08 at 5 13 05 AM" src="https://github.com/user-attachments/assets/dc2cf7c7-c99f-4841-ab9a-8f8d16e4f56b" />


## ✨ Features

* **📄 NLP Resume Parsing:** Upload your PDF resume. The system uses advanced natural language processing to extract your existing technical and soft skills.
* **🎯 12 Supported Career Tracks:** Compare your profile against live industry requirements for Software Engineering, Data Science, Product Management, UX/UI Design, Marketing, and more.
* **⚖️ Intelligent Skill Weighting:** Not all skills are created equal. Using frequency analysis from over 3.3 million LinkedIn job postings, missing skills are categorized into:
  * **🔥 Must Learn** (Appears in 60%+ of postings)
  * **⚠️ Should Learn** (Appears in 30-60% of postings)
  * **💡 Can Learn Later** (Appears in <30% of postings)
* **🧠 Machine Learning Predictions:**
  * **Employability Probability:** A Logistic Regression model predicts your likelihood of landing a job based on a composite score of your skill match percentage and your CGPA.
  * **Career Readiness Segmentation:** A K-Means Clustering algorithm segments your profile against peers into actionable tiers (*High Achiever*, *On Track*, *Needs Focus*).
* **📚 TF-IDF Course Recommender:** Uses Cosine Similarity to recommend the highest-rated, most relevant courses from a massive database of 27,000+ Coursera and Udemy courses to fill your specific skill gaps.

<img width="1668" height="852" alt="Screenshot 2026-05-08 at 5 17 10 AM" src="https://github.com/user-attachments/assets/fb8f9d39-7e9c-4816-ab2d-93e60f0dfde7" />
<img width="1280" height="493" alt="Screenshot 2026-05-08 at 5 17 17 AM" src="https://github.com/user-attachments/assets/e38c6edc-15d7-44ad-bd96-19a6129427df" />


## 🛠️ Tech Stack

* **Frontend / UI:** [Streamlit](https://streamlit.io/)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Logistic Regression, K-Means, TF-IDF Vectorizer, Cosine Similarity)
* **Database:** SQLite (local persistent storage for 27k+ courses and job data)
* **Data Visualization:** Plotly (Interactive Gauges, Radar Charts)
* **Text Extraction:** PyPDF

## 📂 Data Sources
The underlying intelligence of the app is powered by multiple Kaggle datasets processed into a high-speed SQLite database:
1. **LinkedIn Job Postings (3.3M rows):** Used to determine what skills are actually in demand for specific roles and how heavily they should be weighted.
2. **Coursera & Udemy Course Datasets (27,000+ rows):** Processed to match missing skills with actionable learning resources based on difficulty, duration, and ratings.

