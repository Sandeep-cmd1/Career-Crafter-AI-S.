# ================== SECTION 1: Setup & Imports ==================

import google.generativeai as genai
from openai import OpenAI
import fitz  # For PDF reading
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import chromadb
from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor
from tavily import TavilyClient
import json

# Initialize Gemini, OpenAI API clients
gemini_api_key = input("Enter gemini api key: ")
genai.configure(api_key=gemini_api_key)

openai_api_key=input("Enter openai api key: ")
openai_client = OpenAI(api_key=openai_api_key)


# Tavily API client
tavily_client = TavilyClient("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")


# ================== SECTION 2: Core AI Agents ==================

def askAI(prompt, model="gemini-2.5-flash"):
    """Calls Gemini LLM with specified prompt."""
    gemini_client=genai.GenerativeModel(model)
    response = gemini_client.generate_content(prompt)
    return response.text


def web_search_agent(topic):
    """Performs web search on given topic using Tavily."""
    return tavily_client.search(query=topic, max_results=5)


# ================== SECTION 3: Input Handling ==================

# Get user input
user_input = input("Enter your target job description or a job title: ")


#Upload PDF (CV)
root=Tk()
root.withdraw()  # Hide the main tkinter window

filename = askopenfilename(title="Select your CV (PDF)", filetypes=[("PDF files", "*.pdf")])

if filename:
    print(f"Uploaded file: {filename}")

    # Read CV contents
    with fitz.open(filename) as doc:
        cv = "".join(page.get_text() for page in doc)

    print("\n--- Extracted Text ---\n")
    print(cv[:1000])  # Print the first 1000 characters as a preview
else:
    print("No file selected.")

#Removing google colab specific code and added VS code format of file upload using import Tk
"""
# Upload PDF (CV)
from google.colab import files 
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"Uploaded file: {filename}")

# Read CV contents
with fitz.open(filename) as doc:
    cv = "".join(page.get_text() for page in doc)"""


# ================== SECTION 4: Skill Analysis & Embeddings ==================

def skills_listing_agent(data):
    """Extracts categorized skills from given data."""
    skills_listing_prompt = f"""You are an expert in extracting out different types of skills present in data
(or) needed for data provided below. Provide the list of skills with different categories such as technical skills,
leadership skills, soft skills, conginitive skills, language skills, etc. with maximum 6 skills in each category.

data:{data}"""
    return askAI(prompt=skills_listing_prompt)


def chunk_data(data, chunk_size):
    """Splits data into chunks of given word length."""
    words = data.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]



def embedding_agent(data):
  """Generates embeddings using OpenAI"""
  response = openai_client.embeddings.create(
    input=data,
    model="text-embedding-3-small")
  return (response.data[0].embedding)


# ================== SECTION 5: Parallel Skill Processing ==================

with ThreadPoolExecutor() as executor:
    cv_skills, job_skills = executor.map(skills_listing_agent, [cv, user_input])
    cv_skills_chunked, job_skills_chunked = executor.map(chunk_data, [cv_skills, job_skills], [7, 7])

# ================== SECTION 6: Store CV Skills in ChromaDB ==================

chromadb_client = chromadb.Client(Settings(persist_directory='./chromadb_hackathon1'))
collection_db = chromadb_client.get_or_create_collection(name="hackathon1_db")

for i, chunk in enumerate(cv_skills_chunked):
    collection_db.add(
        ids=[f"chunk-{i+1}"],
        documents=[chunk],
        embeddings=[embedding_agent(chunk)]
    )

# ================== SECTION 7: Query Matching ==================

job_skills_embed = embedding_agent(job_skills_chunked)
search_results = collection_db.query(query_embeddings=job_skills_embed, n_results=2)
topmatch_chunks = search_results['documents']
print(topmatch_chunks)


# ================== SECTION 8: Upskilling Plan ==================

def upskilling_guide_agent(topmatch_chunks, job_skills):
    """Identifies skill gaps between CV and target job."""
    skill_mismatch_prompt = f"""
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
    return askAI(prompt=skill_mismatch_prompt)


skills_to_learn = upskilling_guide_agent(topmatch_chunks, job_skills)
print(skills_to_learn)


def training_planning_agent(skills_to_learn):
    """Creates detailed upskilling schedule."""
    training_plan_prompt = f"""
You are an expert in creating a training plan schedule for given list of topics a user wants to learn.
Provide a comprehensive training plan schedule for topics given below.

Output plan in plain text, concise, human readable and it should include detailed sub-categorization of topics,
time taken for each, and other relevant items.
Add title 'Training plan for upskilling of user'

topics:{skills_to_learn}"""
    return askAI(prompt=training_plan_prompt, model="gemini-2.5-pro")


training_plan = training_planning_agent(skills_to_learn)
print(training_plan)

# ================== SECTION 9: Topic Notes, Quiz, Projects ==================

topic = input("""Enter a topic from training plan along with its week title to get lecture notes:
Sample input: Decorator topic in Python Mastery & ML Fundamentals
""")

web_results = web_search_agent(topic)


def lecture_notes_agent(topic, web_results):
    print(f"Generating lecture notes on '{topic}'...")
    lecture_notes_prompt = f"""You are an expert tutor in the field of a topic given below. On given topic, generate a detailed
lecture notes with concise definitions, clear concept explanations, examples, case studies or use cases and
other educational information.

Additionally, refer to each 'content' in web_results data, extract any new information related to given topic and add it
logically, reasonably, organically and meaningfully in lecture notes along with citations or reference to 'url' associated
to that 'content' in web_results data.

Output should be concise, in plain text and human readable.

topic:{topic}
web_results:{web_results} """
    return askAI(prompt=lecture_notes_prompt, model="gemini-2.5-pro")


lecture_notes = lecture_notes_agent(topic, web_results)
print(lecture_notes)

# Quiz/Assignments/Interview agents
def quizzing_agent(lecture_notes):
    quiz_prompt = f"""You are an expert quiz master. Generate 10 quiz questions based on notes given below.
Questions should be in multiple chhoice format where a question is given 4 options and only one option
is correct out of 4. Give answer key of correct options for all questions at the end of 10 quiz questions.

notes:{lecture_notes}"""
    return askAI(prompt=quiz_prompt)


def assgn_pjideas_agent(lecture_notes):
    assgn_pjs_prompt = f"""You are an expert tutor on notes given below. Generate 3 coding assignment questions
based on give notes, 2 questions should be of basic level and 1 question of advanced level.
Give detailed solutions for 3 questions after listing out these 3 questions.

Later, under title 'Projects to try out!', suggest three real-world projects ideas related to notes which user can try out.

In the end, display 5 frequently asked oral interview questions relatd to notes along with their brief and optimized answers.

Ouput should be concise, in plain text and human readable.

notes:{lecture_notes}"""
    return askAI(prompt=assgn_pjs_prompt)


def faqs_of_interview_agent(lecture_notes):
    faqs_prompt = f"""You are an expert tutor and interviewer on notes given below.
Provide 10 frequently asked oral interview questions relatd to notes along with their optimized answers.

Ouput should be concise, in plain text and human readable.

notes:{lecture_notes}"""
    return askAI(prompt=faqs_prompt)


def other_queries_agent(lecture_notes):
    chat_history = [lecture_notes]
    while True:
        query = input("Please enter your message ('exit' ends chat): ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        chat_history.append(f"USER: {query}")
        response = askAI(prompt=chat_history)
        print(response)
        chat_history.append(f"AI: {response}")


# ================== SECTION 10: Agent Selection ==================

user_query = input("Enter a query about selected topic such as - quiz, assignment, interview questions, others: ")

def assigning_agent(user_query):
    routing_prompt = f"""
You are a routing assistant.

Based on the user_query, identity the appropriate agent out of all available agents to handle the user request

Available Agents:
- Quiz agent
- Assignment, project ideas agent
- FAQs of interview agent
- Other queries agent

Response **ONLY** with name of the appropriate agent

user_query : {user_query}"""
    return askAI(prompt=routing_prompt)


def handle_agents(agent, lecture_notes):
    print(f"Redirecting you to '{agent}'...")
    match agent:
        case "Quiz agent":
            print(quizzing_agent(lecture_notes))
        case "Assignment, project ideas agent":
            print(assgn_pjideas_agent(lecture_notes))
        case "FAQs of interview agent":
            print(faqs_of_interview_agent(lecture_notes))
        case "Other queries agent":
            other_queries_agent(lecture_notes)

agent = assigning_agent(user_query)
handle_agents(agent, lecture_notes)


# ================== SECTION 11: Final CV & LinkedIn Content ==================

def cv_template_agent(training_plan):
    cv_template_prompt = f"""You are an expert in Resume or CV making. Generate Resume or CV content to reflect
all knwoledge, skills and experience a user gained by learning all topics in training_plan give below.

Output can have sections and sub-sections such as 'Summary', 'Technical Skills', 'Functional skills',
'Engineering experience with projects and results', 'Leadership experience with projects and results', 'Conclusion' and others.
You can also add additional empty sections or sub-sections which are relevant to training_plan as a template for others.

Output should be very concise, Application Tracking System (ATS) friendly, maximum of 30 lines, in plain text and human readable.

training_plan:{training_plan}"""
    return askAI(prompt=cv_template_prompt)


def linkedin_post_agent(training_plan):
    linkedin_post_prompt = f"""You are a social media strategist and content generator. Create tailored post for LinkedIn of
maximum 20 lines that experesses the experience of a user who gained knowledge, skills and expertise by learning all topics
in training_plan give below.

The tailored post should sound professional, insightful, educational and value-driven that focus on key takeaways in bullet points.

📌 Target Audience: [Tech Professionals, Entrepreneurs, Leaders, Students]
Add 5 relevant hashtags at the end.

🔍  training_plan:{training_plan}"""
    return askAI(prompt=linkedin_post_prompt)


final_input = input("Upskilling on training plan completed? (y/n): ").strip().lower()
if final_input == 'y':
    print("\n\n!!!CONGRATULATIONS!!!!\n\nGenerating a CV content template and a linkedin_post...")
    with ThreadPoolExecutor() as executor:
        agents = [cv_template_agent, linkedin_post_agent]
        output = list(executor.map(lambda fn: fn(training_plan), agents))
        print(f"\n\nCV_Template:\n{output[0]}\n\n")
        print(f"Linkedin_Post:\n{output[1]}\n\n")
elif final_input == 'n':
    print("Please carry on with your upskilling journey.")
else:
    print("Invalid input. Please enter 'y' or 'n'.")
