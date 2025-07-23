#CAREER CRAFTER AI APP

# ================== SECTION: IMPORTS ==================

import streamlit as st
import google.generativeai as genai
import fitz
from concurrent.futures import ThreadPoolExecutor
from tavily import TavilyClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# ================== SECTION: API KEYS & CONFIG ==================

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
tavily_client = TavilyClient(st.secrets["TAVILY_API_KEY"])

# ================== SECTION: Streamlit UI Setup ==================

st.set_page_config(page_title="AI Upskilling App", layout="wide")

# Center-aligned Title & Subheader
st.markdown("""
    <div style='text-align:center;'>
        <h1>🚀 Career Crafter AI</h1>
        <h3>AI - Empowering Human Career and Not Replacing It</h3>
    </div>
    <br>
""", unsafe_allow_html=True)

# ================== SECTION: Session State Initialization ==================

# 🧠 Streamlit Tip:
# st.session_state is used to persist variables (not re-set) across code reruns triggered by user actions
# st.cache_data is used to cache expensive computations or API calls and prevent redundant execution for same arguments of function.

def initialize_session():
    defaults = {
        "user_input": "",
        "uploaded_file": None,
        "cv": "",
        "topic": "",
        "user_query": "",
        "final_input": "No",
        "skills_saved_to_db": False,
        "cv_skills": "",
        "job_skills": "",
        "cv_skills_chunked": [],
        "job_skills_chunked": [],
        "job_embeddings": [],
        "topmatch_chunks": []
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session()

# ================== SECTION: Helper Functions ==================

# Skills Listing Agent: Extracts categorized skills from text
def skills_listing_agent(data):
    prompt = f"""You are an expert in extracting out different types of skills present in data
    (or) needed for data provided below. Provide the list of skills with different categories such as technical skills,
    leadership skills, soft skills, conginitive skills, language skills, etc. with maximum 6 skills in each category.

    data:{data}"""
    return askAI(prompt)

# Chunker: Splits text into chunks of specified word length (chunk_size)
def chunk_data(data, chunk_size):
    words = data.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

#Converts data to documents for langchain_chromadb embedding
def convert_to_documents(chunks):
    return [Document(page_content=chunk) for chunk in chunks]

# Upskilling guiding Agent: Suggests required extra skills for user to be eligible for a target job based on user's CV skills
@st.cache_data(show_spinner="Generating required skills to learn...")
def upskilling_guide_agent(topmatch_chunks, job_skills):
    prompt = f"""
    You are an expert in creating following two lists related to current skills of an user and required skills of a target job:
    1) SKills a user already have that matches with the required skills of a target job
    2) Skills a user is yet to learn to perfectly match user's skills set with the required skills of a target job.
    The context to understand current skills of an user is given below as 'context'. The required skills of a target job
    are given below as 'target job skills'.

    But output ONLY below list of skills with sub-skills under each skill using concise words:
    --> Skills a user is yet to learn to perfectly match user's skills set with the required skills of a target job.
    Add a title 'User should upskill on following skills to be eligible for the target job description/ job title"

    context: {topmatch_chunks}
    target job skills: {job_skills}"""
    return askAI(prompt)

# Training Plan Generator Agent: Creates a detailed training plan for upskilling
@st.cache_data(show_spinner="Generating training plan for upskilling...")
def training_planning_agent(skills_to_learn):
    prompt = f"""
    You are an expert in creating a training plan schedule for given list of topics a user wants to learn.
    Provide a comprehensive training plan schedule for topics given below.

    Output plan in plain text, concise, human readable and it should include detailed sub-categorization of topics,
    time taken for each, and other relevant items.
    Add title 'Training plan for upskilling of user'

    topics:{skills_to_learn}"""
    return askAI(prompt)

# Lecture Notes Agent: Generates lecture notes on a given topic by combining latest web data
@st.cache_data(show_spinner="Generating lecture notes on given topic...")
def lecture_notes_agent(topic, web_results):
    prompt = f"""You are an expert tutor in the field of a topic given below. On given topic, generate a detailed
    lecture notes with concise definitions, clear concept explanations, examples, case studies or use cases and
    other educational information.

    Additionally, refer to each 'content' in web_results data, extract any new information related to given topic and add it
    logically, reasonably, organically and meaningfully in lecture notes along with citations or reference to 'url' associated
    to that 'content' in web_results data.

    Output should be concise, in plain text and human readable.

    topic:{topic}
    web_results:{web_results} """
    return askAI(prompt)

# Quiz Generator Agent: Creates multiple choice questions quiz (answers) based on lecture notes
@st.cache_data(show_spinner="Generating quiz (answers) based on lecture notes...")
def quizzing_agent(lecture_notes):
    prompt = f"""You are an expert quiz master. Generate 10 quiz questions based on notes given below.
    Questions should be in multiple chhoice format where a question is given 4 options and only one option
    is correct out of 4. Give answer key of correct options for all questions at the end of 10 quiz questions.

    notes:{lecture_notes}"""
    return askAI(prompt)

# Assignment (solutions) and Project Ideas Generator Agent based on lecture notes
@st.cache_data(show_spinner="Generating assignment (solutions) and project ideas based on lecture notes...")
def assgn_pjideas_agent(lecture_notes):
    prompt = f"""You are an expert tutor on notes given below. Generate 3 coding assignment questions
    based on give notes, 2 questions should be of basic level and 1 question of advanced level.
    Give detailed solutions for 3 questions after listing out these 3 questions.

    Later, under title 'Projects to try out!', suggest three real-world projects ideas related to notes which user can try out.

    Ouput should be concise, in plain text and human readable.

    notes:{lecture_notes}"""
    return askAI(prompt)

# FAQ Generator Agent: Generate common interview questions & answers based on lecture notes
@st.cache_data(show_spinner="Generating frequently asked questions & answers based on lecture notes...")
def faqs_of_interview_agent(lecture_notes):
    prompt = f"""You are an expert tutor and interviewer on notes given below.
    Provide 10 frequently asked oral interview questions relatd to notes along with their optimized answers.

    Ouput should be concise, in plain text and human readable.

    notes:{lecture_notes}"""
    return askAI(prompt)

# CV Template Generator Agent: Prepares a concise, ATS-friendly CV content based on the finished training plan
@st.cache_data(show_spinner="Generating CV template for upskilled items...")
def cv_template_agent(training_plan):
    prompt = f"""You are an expert in Resume or CV making. Generate Resume or CV content to reflect
    all knwoledge, skills and experience a user gained by learning all topics in training_plan give below.

    Output can have sections and sub-sections such as 'Summary', 'Technical Skills', 'Functional skills',
    'Engineering experience with projects and results', 'Leadership experience with projects and results', 'Conclusion' and others.
    
    You can also add additional empty sections or sub-sections which are relevant to training_plan as a template for others.

    Output should be very concise, Application Tracking System (ATS) friendly, maximum of 30 lines, in plain text and human readable.

    training_plan:{training_plan}"""
    return askAI(prompt)

# LinkedIn Post Generator Agent: Creates a professional post to showcase upskilling journey
@st.cache_data(show_spinner="Generating linkedin post to share about upskilled items...")
def linkedin_post_agent(training_plan):
    prompt = f"""You are a social media strategist and content generator. Create tailored post for LinkedIn of
    maximum 20 lines that experesses the experience of a user who gained knowledge, skills and expertise by learning all topics
    in training_plan give below.

    The tailored post should sound professional, insightful, educational and value-driven that focus on key takeaways in bullet points.

    Target Audience: [Tech Professionals, Entrepreneurs, Leaders, Students]
    Add 5 relevant hashtags at the end.

    training_plan:{training_plan}"""
    return askAI(prompt)

# ================== SECTION: Layout Columns ==================
col1, col2, col3 = st.columns([1.2, 2, 1.2])

# ================== SECTION: Column 1 - Inputs ==================
with col1:
   
    st.session_state.user_input = st.text_input("🎯 Enter your target job description or job title:", value=st.session_state.user_input)
    st.session_state.uploaded_file = st.file_uploader("📄 Upload your CV (PDF)", type="pdf")

    # If both job title/ description and CV are provided, process the PDF
    if st.session_state.uploaded_file and st.session_state.user_input:
        with fitz.open(stream=st.session_state.uploaded_file.read(), filetype="pdf") as doc:
            # 🔍 Extract text from all pages of the uploaded CV PDF
            st.session_state.cv = "".join(page.get_text() for page in doc)
        st.success("✅ CV uploaded and processed!")
        # 📑 Show a preview of extracted CV text (first 1000 characters)
        st.text_area("📑 Preview of CV Text", st.session_state.cv[:1000], height=400)
    else:
        # 🚫 Stop execution if input or uploaded file is missing
        st.stop()

# ================== SECTION: Column 2 - Processing ==================
with col2:

    # Function to interact with Gemini model
    def askAI(prompt, model="gemini-2.5-flash"):
        gemini_client = genai.GenerativeModel(model)
        response = gemini_client.generate_content(prompt)
        return response.text

    # Web search using Tavily API
    @st.cache_data
    def web_search_agent(topic):
        return tavily_client.search(query=topic, max_results=5)

    with st.spinner("Extracting and chunking skills from CV and job description ..."):
        #Extract skills from CV and Job description (if not already done)
        if not st.session_state["cv_skills"] or not st.session_state["job_skills"]:
            with ThreadPoolExecutor() as executor:
                futures = {}
                if not st.session_state["cv_skills"]:
                    # Extract skills from CV using skills_listing_agent
                    futures["cv_skills"] = executor.submit(skills_listing_agent, st.session_state.cv)
                if not st.session_state["job_skills"]:
                    # Extract skills from target job description or title using skills_listing_agent
                    futures["job_skills"] = executor.submit(skills_listing_agent, st.session_state.user_input)
                for key, future in futures.items():
                    st.session_state[key] = future.result() #In case of 'executor.submit', need '.result()' to extract value of object future

        # Chunk the extracted skills (CV & Job description/ title) text into groups of 7 words
        if not st.session_state["cv_skills_chunked"] or not st.session_state["job_skills_chunked"]:
            with ThreadPoolExecutor() as executor:
                cv_skills_chunked, job_skills_chunked = list(executor.map(
                    chunk_data,
                    [st.session_state["cv_skills"], st.session_state["job_skills"]],
                    [7, 7]
                ))
                st.session_state.cv_skills_chunked = cv_skills_chunked
                st.session_state.job_skills_chunked = job_skills_chunked

    # Initialize langchain chromaDB vector database client
    vectordb = Chroma(
        collection_name="hackathon1_db",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chromadb_hackathon1"
    )

    # Save CV skill chunks to vector database (if not already done)
    if not st.session_state.skills_saved_to_db:
        with st.spinner("💾 Saving CV skill chunks to vector DB..."):
            docs = convert_to_documents(st.session_state.cv_skills_chunked)
            vectordb.add_documents(docs) #Embedding and storing docs in vector database
            st.session_state.skills_saved_to_db = True

    # Convert job skills chunks into documents and embed them 
    if not st.session_state.get("job_embeddings"):
        with st.spinner("Embedding job skills..."):
            job_docs = convert_to_documents(st.session_state.job_skills_chunked)
            job_embeddings = OpenAIEmbeddings().embed_documents([doc.page_content for doc in job_docs])
            st.session_state.job_embeddings = job_embeddings

    #Do vector semantic search of job skills embeddings in vector database to extract top matches
    if not st.session_state.get("topmatch_chunks"):
        with st.spinner("Performing vectors similarity search and extracting top match results..."):
            def search(embed):
                return vectordb.similarity_search_by_vector(embed, k=2)
            with ThreadPoolExecutor() as executor:
                all_matches = list(executor.map(search, st.session_state.job_embeddings))
            # Flatten all_matches and extract top match results
            flattened_results = [doc for sublist in all_matches for doc in sublist]
            st.session_state.topmatch_chunks = [doc.page_content for doc in flattened_results]

    # Generate list of skills user needs to learn to achieve target job
    skills_to_learn = upskilling_guide_agent(st.session_state.topmatch_chunks, st.session_state.job_skills)
    st.subheader("📈 Additional skills you need to learn for target job:")
    with st.expander("Click to view"):
        st.markdown(skills_to_learn)

    #Generate personalized training plan for upskilling
    training_plan = training_planning_agent(skills_to_learn)
    st.subheader("📅 Training Plan for upskilling")
    with st.expander("Click to view"):
        st.markdown(training_plan)

    # Enter a specific topic from the training plan to get further study materials
    st.session_state.topic = st.text_input("📘 Enter a topic from training plan:",
        placeholder="e.g. Decorator topic in Python Mastery & ML Fundamentals", value=st.session_state.topic)

    if st.session_state.topic:
        # Search the web to extract and append latest info with citations & refernces into lecture notes
        web_results = web_search_agent(st.session_state.topic)

        #Generate lecture notes on a given topic from training plan
        lecture_notes = lecture_notes_agent(st.session_state.topic, web_results)
        st.subheader(f"📈 Lecture notes on {st.session_state.topic}:")
        with st.expander("Click to view"):
            st.markdown(lecture_notes)

        # Let user ask for quiz, assignments, or interview questions related to lecture notes
        st.session_state.user_query = st.text_input("🧠 Ask for quiz or assignments or interview questions:", value=st.session_state.user_query)

        if st.session_state.user_query:
            # Assigning Agent: Use routing logic to determine the best suited agent for user query
            def assigning_agent(query):
                prompt = f"""
                You are a routing assistant.

                Based on the user_query, identity the appropriate agent out of all available agents to handle the user request

                Available Agents:
                - Quiz agent
                - Assignment, project ideas agent
                - FAQs of interview agent

                Response **ONLY** with name of the appropriate agent

                user_query : {query}"""
                return askAI(prompt)

            #Calling Assigning Agent based on user query
            agent = assigning_agent(st.session_state.user_query)
            st.write(f"🔄 Redirecting to: **{agent}**")

            # Route to appropriate content generation agent
            match agent:
                case "Quiz agent":
                    #Generates quiz with answers based on lecture notes
                    st.subheader("Quiz with answers on lecture notes:")
                    with st.expander("Click to view"):
                        st.markdown(quizzing_agent(lecture_notes))
                case "Assignment, project ideas agent":
                    #Generates assignments (solutions) and project ideas based on lecture notes
                    st.subheader("Assignments (solutions) & Project ideas:")
                    with st.expander("Click to view"):
                        st.markdown(assgn_pjideas_agent(lecture_notes))
                case "FAQs of interview agent":
                    #Generates frequently asked interview questions (answers) based on lecture notes
                    st.subheader("Interview Questions & Answers:")
                    with st.expander("Click to view"):
                        st.markdown(faqs_of_interview_agent(lecture_notes))

# ================== SECTION: Column 3 - Final CV & LinkedIn ==================
with col3:
    
    # ✅ Checkbox here is to confirm if upskilling is completed
    # Set session_state.final_input based on checkbox state (Yes/No)

    st.session_state.final_input = "Yes" if st.checkbox("✅ Upskilling completed?", value=(st.session_state.final_input == "Yes")) else "No"

    # If user has completed upskilling, this generates CV template and LinkedIn post
    if st.session_state.final_input == "Yes":
        with st.spinner("Generating customized CV template & Linkedin post..."):
            st.subheader("🎓 CV & LinkedIn Post")

            # Run both agents (CV and LinkedIn functions) in threads
            with ThreadPoolExecutor() as executor:
                output = list(executor.map(lambda fn: fn(training_plan), [cv_template_agent, linkedin_post_agent]))

            # Display generated CV template
            st.subheader("📝 CV Template")
            with st.expander("Click to view"):
                st.markdown(output[0])

            # Display generated LinkedIn post
            st.subheader("🔗 LinkedIn Post")
            with st.expander("Click to view"):
                st.markdown(output[1])
    else:
        # Below message shown if upskilling not yet marked complete
        st.info("Continue your upskilling journey!")