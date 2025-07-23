
APP DESCRIPTION: 🚀 Career Crafter AI

Career Crafter AI is a Streamlit-based application that uses powerful LLMs (Gemini, OpenAI) and web tools (Tavily, ChromaDB) to help job seekers identify skill gaps, generate personalized upskilling plans, generate study materials (notes, quiz, assigments, project ideas and interview FAQs) and craft a tailored CV & LinkedIn post—all powered by AI.


WHAT THIS APP DOES:

    - Analyze your current CV and extract your skills

    - Extract key skills from your target job description [time taken:15-20 sec]

    - Identify and generate missing skills (skill gap analysis) [time taken:10-15 sec]

    - Build a personalized training plan [time taken:15-20 sec]

    - Generate lecture notes appended with latest info from the web (adds citations & references) [time taken:20-25 sec]

    - Create quizzes (answers), assignments (solutions), project ideas and interview FAQs (answers) [time taken:10-15 sec]

    - Generate a concise, ATS-friendly CV template on upskilled items [time taken:10-15 sec for both CV & linkedin post]

    - Create a professional LinkedIn post to showcase your learning journey 


HOW APP WORKS:

    Input1: Enter your target job title or description

    Input2: Upload your CV 

    Skill Extraction: In background, the app extracts and embeds skills from both CV and job description

    Skill Gap Analysis: In background, the app performs semantic similarity engine (langchain ChromaDB + embeddings) 
    
    Ouput1: Creates and displays skills you should learn to be a perfect fit for target job description

    Output2: Creates and displays a tailored upskilling roadmap with subtopics and timelines

    Input3: Enter a topic from training plan

    Output3: Creates and displays latest lecture notes with inputs from LLM and web search (adds citations & references)
    
    Input4: Ask the app for quiz, assignments and interview FAQs on lecture notes
    
    Output4: Creates and displays quiz (answers) or assignments (solutions)/ project ideas or interview FAQs (answers)

    Final Output5: Once upskilled, checkmark completed option and the app generates a new CV template and a professional LinkedIn post to showcase upskilled results.

SAMPLE INPUT:

    Target Job: Machine Learning Engineer with Cloud Experience

    CV Upload: A PDF resume listing Python, SQL, and basic ML skills

SAMPLE OUTPUT:

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

    Interview Questions & Answers Excerpt:

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

TECH STACK:

    Frontend: Streamlit

    LLMs: Gemini (Google)

    web Search: Tavily API

    Embedding & Vector DB: Langchain-openai, Langchain-chromadb

    Parallel Execution: Python ThreadPoolExecutor

REQUIREMENTS:

    Streamlit

    google-generativeai

    langchain

    PyMuPDF (fitz)

    langchain-openai

    langchain-community

    langchain-chroma

    tavily-python