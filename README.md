<<<<<<< HEAD
Access this web-app via https://skillgap-analyzer.streamlit.app/
=======
# SkillBridge: AI-Powered Skill Gap Analyzer 🚀

SkillBridge is an advanced, data-driven career readiness platform built to bridge the gap between student resumes and industry requirements. By leveraging Machine Learning, NLP, and massive real-world datasets, SkillBridge analyzes your profile, identifies missing skills, and recommends hyper-personalized courses to make you job-ready.

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

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/khan-abd/skill-gap-analyzer.git
cd skill-gap-analyzer
```

### 2. Install dependencies
Make sure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```
The app will automatically open in your default web browser at `http://localhost:8501`.

## ☁️ Deployment
This application is fully optimized for deployment on **Streamlit Community Cloud**. The SQLite database is pre-built and included in the repository to ensure lightning-fast boot times without the need to process the massive CSV files during runtime.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Built with ❤️ using Python and Streamlit.*
>>>>>>> 38595b6 (Add README.md)
