
APP DESCRIPTION: 🚀 Career Crafter AI

Career Crafter AI is a Streamlit-based application that uses powerful LLMs (Gemini, OpenAI) and web tools (Tavily, ChromaDB) to help job seekers identify skill gaps, generate personalized upskilling plans, and craft a tailored CV & LinkedIn post—all powered by AI.


🧠 What This App Does

    📄 Analyze your current CV 

    🎯 Extract key skills from your target job description

    📈 Identify missing skills (skill gap analysis)

    📅 Build a personalized training plan

    📘 Generate lecture notes appended with latest info from the web

    ❓ Create quizzes, assignments, project ideas and interview FAQs

    📝 Generate a concise, ATS-friendly CV template on upskilled items

    🔗 Create a professional LinkedIn post to showcase your learning journey


⚙️ How It Works

    Input: Upload your CV and enter your target job title or description.

    Skill Extraction: In background, the app extracts and embeds skills from both CV and job description.

    Skill Gap Analysis: The app performs semantic similarity engine (ChromaDB + embeddings) and displays unmatched skills.

    Training Plan: The app creates and displays a tailored upskilling roadmap with subtopics and timelines.

    Topic Study: Choose a topic from training plan and app displays latest lecture notes. Further you can ask the app for quizzes, assignments, and FAQs on lecture notes, it delivers.

    Final Output: Once upskilled, checkmark completed option and the app generates a new CV template and a professional LinkedIn post to showcase upskilled results.

📥 Sample Input

    Target Job: Machine Learning Engineer with Cloud Experience

    CV Upload: A PDF resume listing Python, SQL, and basic ML skills

📤 Sample Output

    Additional Skills to Learn:

        MLOps with Kubeflow

        Cloud Deployment (GCP/AWS)

        Model Optimization & Monitoring

        Data Pipeline Automation

    Training Plan Excerpt:

        Training Plan for Upskilling of User

        1. MLOps: 2 weeks  

        - Kubeflow pipelines

        - CI/CD for ML models  

    Lecture Notes Excerpt:

        Technical SEO Basics: Lecture Notes

        1. Introduction to Technical SEO

        Definition: ....

        Why it Matters: ....

        2. Core Concepts: Crawling and Indexing....

    Interview Questions Excerpt:
        Here are 10 frequently asked oral interview questions related to Technical SEO, along with optimized answers:

        1. What is Technical SEO and why is it crucial for a successful SEO strategy?

        Answer: Technical SEO involves optimizing a website's infrastructure and backend elements to help search engines efficiently find, crawl, understand, and index pages. It's crucial because without a strong technical foundation, even high-quality content and link-building efforts may not yield desired results, as search engines might not be able to access or interpret the content effectively.

        2. Explain the concepts of crawling and indexing in the context of search engines. How does Technical SEO influence these processes?

        Answer: Crawling is when search engine "crawlers" or "spiders......

    Generated CV Snippet:

        Summary: Hands-on Machine Learning Engineer with newly acquired expertise in MLOps, Cloud Deployment (GCP), and Data Engineering...

    LinkedIn Post:

        "Just wrapped up a 6-week upskilling journey into advanced ML and Cloud MLOps! 🚀 Highlights: ✅ GCP Deployment, ✅ Kubeflow Pipelines...
        #MachineLearning #CareerGrowth #MLOps #CloudEngineering #AI"

🧰 Tech Stack

    Frontend: Streamlit

    LLMs: Gemini (Google)

    web Search: Tavily API

    Embedding & Vector DB: OpenAI + ChromaDB

    Parallel Execution: Python ThreadPoolExecutor

🛠 Requirements

    Streamlit

    google-generativeai

    openai

    PyMuPDF (fitz)

    chromadb

    tavily-python