import streamlit as st
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from io import BytesIO
import easyocr
import os
import urllib.parse
from PIL import Image
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import asyncio
import glob
import time

# --- API Keys ---
GROQ_API_KEY = "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" # Replace with your actual key or use st.secrets
GOOGLE_API_KEY = "AIzaSyB5dlbtndihCliWB1GCXoZJaTwVYXidBVg" # Replace with your actual key or use st.secrets

# --- OCR and LLM Setup ---
reader = easyocr.Reader(['en'])
chat_model = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a friendly and knowledgeable assistant who provides clear and engaging information about SIMATS engineering college. You have access to official college documents.

**When responding, follow these guidelines:**
1. **Humanize your tone**: Write as if you're explaining to a friend—use contractions ("you'll", "we've"), occasional humor, and relatable examples.
    - *Example*: "Think of this like choosing between Netflix plans—but for courses!"
2. **Structure for clarity**:
    - Start with a **1-sentence summary**.
    - Use **bullet points** for key details.
    - Add a **real-world scenario** or analogy if helpful.
3. **Adapt to the user**:
    - If the question is formal (e.g., about deadlines), keep it professional.
    - If casual (e.g., "What's campus life like?"), be conversational and don't tell I don't know or there is no information from the provided document.

**Avoid**:
- Robotic phrases like "Based on the provided context..."
- Overloading with jargon. Simplify technical terms.
"""),
    HumanMessagePromptTemplate.from_template("""**Context from college documents**:
{context}

**Question**: {question}

**Task**: Craft a response that:
1. Starts with a **hook** (e.g., "Great question!" or "Let's break this down...").
2. Answers *all* parts of the question with **blended context** (no copy-paste).
3. Uses **examples** (e.g., "For instance, last year a student...").
4. Ends with a helpful note or suggestion.
""")
])

# --- WebLoader with Image OCR ---
class WebLoaderWithImageOCR(WebBaseLoader):
    def __init__(self, web_paths, *args, **kwargs):
        super().__init__(web_paths, *args, **kwargs)
        self.ocr_reader = reader
        self.requests_kwargs = {"headers": {"User-Agent": "SIMATS-Chatbot/1.0 (Contact: testingpurposebuddy@gmail.com)"}}

    def _check_if_image(self, url, content_type=None):
        if content_type:
            is_image = content_type.startswith('image/')
            is_gif = content_type == 'image/gif'
            return is_image and not is_gif
        parsed_url = urllib.parse.urlparse(url)
        _, ext = os.path.splitext(parsed_url.path)
        return ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']

    def _extract_text_from_image(self, img_url):
        try:
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if not self._check_if_image(img_url, content_type):
                    return ""
                img = Image.open(BytesIO(response.content))
                import numpy as np
                result = self.ocr_reader.readtext(np.array(img))
                extracted_text = " ".join(text for _, text, prob in result if prob > 0.5)
                return extracted_text.strip()
            return ""
        except Exception as e:
            print(f"Error processing image {img_url}: {e}")
            return ""

    def load(self):
        import numpy as np
        docs = super().load()
        for i, doc in enumerate(docs):
            soup = bs4.BeautifulSoup(doc.page_content, 'html.parser')
            img_texts = []
            for img in soup.find_all('img'):
                img_url = img.get('src', '')
                if not img_url or img_url.startswith('data:'):
                    continue
                if not img_url.startswith(('http://', 'https://')):
                    base_url = doc.metadata.get('source', '')
                    img_url = urllib.parse.urljoin(base_url, img_url)
                img_text = self._extract_text_from_image(img_url)
                if img_text:
                    img_texts.append(f"Image text: {img_text}")
            if img_texts:
                combined_text = doc.page_content + "\n\n" + "\n".join(img_texts)
                docs[i].page_content = combined_text
        return docs

PDF_FOLDER = "data/pdfs"

@st.cache_resource(show_spinner=False) # Suppress default Streamlit spinner
def load_and_process_documents():
    # Simulate loading delay for demonstration of custom loader
    time.sleep(2)

    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created folder: {PDF_FOLDER}. Please add PDF files to this folder.")

    pdf_files = glob.glob(f"{PDF_FOLDER}/*.pdf")
    if not pdf_files:
        print(f"No PDF files found in {PDF_FOLDER}. Proceeding with website data only.")

    website_urls = [
        "https://simatsengineering.com/",
        "https://simatsengineering.com/simats-accreditations",
        "https://collegedunia.com/college/56310-saveetha-school-of-engineering-sse-chennai",
        "https://sites.google.com/saveetha.com/dmc/list-of-faculty",
        "https://simatsengineering.com/profile",
        "https://in.linkedin.com/school/saveetha-school-of-engineering/",
        "https://simatsengineering.com/incubation-centre",
        "https://simatsengineering.com/research-new",
        "https://simatsengineering.com/infrastructure",
        "https://simatsengineering.com/best-practices",
        "https://www.saveetha.com/mediacoverage",
        "https://www.saveetha.com/ins",
        "https://www.saveetha.com/policies",
        "https://simatsengineering.com/computer",
        "https://simatsengineering.com/cse-programs",
        "https://simatsengineering.com/cse-facilities",
        "https://simatsengineering.com/cse-research",
        "https://simatsengineering.com/office-of-international-affairs",
        "https://simatsengineering.com/cse-faculty",
        "https://simatsengineering.com/ece-1",
        "https://simatsengineering.com/eee",
        "https://simatsengineering.com/biomedical",
        "https://simatsengineering.com/new-page-28",
        "https://simatsengineering.com/bioinformatics-1",
        "https://simatsengineering.com/energy-and-environmental",
        "https://simatsengineering.com/mechanical",
        "https://simatsengineering.com/pageit",
        "https://simatsengineering.com/agriculture",
        "https://simatsengineering.com/ai-ml",
        "https://simatsengineering.com/automobile",
        "https://simatsengineering.com/biotechnology",
        "https://simatsengineering.com/civil",
        "https://simatsengineering.com/ai-ds",
        "https://simatsengineering.com/admissions",
        "https://simatsengineering.com/placement",
        "https://simatsengineering.com/iic",
        "https://simatsengineering.com/news-1",
        "https://simatsengineering.com/contact-us",
        "https://simatsengineering.com/collaborations",
        "https://simatsengineering.com/news",
        "https://www.saveetha.com/sports-and-cultural-facilities",
    ]

    pdf_docs = [doc for pdf in pdf_files for doc in PyPDFLoader(pdf).load()]
    website_docs = []
    for url in website_urls:
        try:
            loader = WebLoaderWithImageOCR(web_paths=(url,))
            website_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {url}: {e}")

    all_docs = pdf_docs + website_docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
    all_splits = text_splitter.split_documents(all_docs)

    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents=all_splits)
    retriever = vector_store.as_retriever()

    print(f"Total sub-documents created: {len(all_splits)}")
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def async_retrieve_documents(question, retriever):
    return format_docs(await asyncio.to_thread(retriever.invoke, question))

async def create_augmented_answer(question, retriever):
    context = await async_retrieve_documents(question, retriever)
    full_context = {"context": context, "question": question}
    response = await asyncio.to_thread(chat_model.invoke, chat_template.format_messages(**full_context))
    return response.content

async def get_college_info(question, retriever):
    try:
        return await create_augmented_answer(question, retriever)
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_college_info_sync(question, retriever):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(get_college_info(question, retriever))
    loop.close()
    return result

def typewriter_effect(text, placeholder):
    """Create a typewriter effect for text"""
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text + "▌")
        time.sleep(0.02)  # Adjust speed as needed
    placeholder.markdown(displayed_text)

# --- Streamlit App Setup ---
st.set_page_config(
    page_title="SIMATS Chatbot",
    page_icon="🎓", # Using an emoji here as direct icon integration can be complex without FontAwesome
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Dark Blue & Milk White, Glassmorphism, and Loading Screen ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stApp"] {
        background: #1A202C; /* Dark Blue Background */
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
        overflow-x: hidden;
    }
    
    [data-testid="stAppViewContainer"] {
        background: rgba(255, 255, 255, 0.05); /* Slightly transparent white for glassmorphism */
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        max-width: 100vw;
        margin: 0;
        padding: 0;
        min-height: 100vh;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #2A3B4C, #1A202C); /* Darker blue gradient */
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px 30px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .header-title {
        color: #F8F8F8; /* Milk White */
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .header-subtitle {
        color: rgba(248, 248, 248, 0.8); /* Slightly transparent Milk White */
        font-size: 1.1rem;
        margin-top: 5px;
        font-weight: 400;
    }
    
    /* Chat Container */
    .chat-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
        height: calc(100vh - 120px);
        overflow-y: auto;
    }
    
    /* Message Styling */
    .stChatMessage {
        padding: 16px 20px !important;
        margin: 12px 0 !important;
        border-radius: 18px !important;
        max-width: 85% !important;
        word-wrap: break-word !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        animation: slideIn 0.3s ease-out !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #364F6B, #2A3B4C) !important; /* Dark Blue Shades */
        color: #F8F8F8 !important; /* Milk White */
        margin-left: auto !important;
        margin-right: 0 !important;
        border-radius: 18px 18px 4px 18px !important;
    }
    
    [data-testid="chat-message-assistant"] {
        background: #F8F8F8 !important; /* Milk White */
        color: #1A202C !important; /* Dark Blue */
        margin-left: 0 !important;
        margin-right: auto !important;
        border-radius: 18px 18px 18px 4px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Input Styling */
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.05) !important; /* Transparent white */
        backdrop-filter: blur(20px) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 20px !important;
        position: sticky !important;
        bottom: 0 !important;
        z-index: 100 !important;
    }
    
    .stChatInput > div {
        background: #F8F8F8 !important; /* Milk White */
        border-radius: 25px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stChatInput input {
        background: transparent !important;
        border: none !important;
        color: #1A202C !important; /* Dark Blue */
        font-size: 16px !important;
        padding: 12px 20px !important;
        font-weight: 400 !important;
    }
    
    .stChatInput input::placeholder {
        color: #4A5568 !important; /* A slightly lighter dark blue */
        opacity: 0.8 !important;
    }
    
    /* Custom Loading Overlay */
    #custom-loader-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #1A202C 0%, #2A3B4C 100%); /* Dark blue gradient */
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
        animation: fadeIn 0.5s ease-out;
    }
    
    #custom-loader-overlay.fade-out {
        animation: fadeOut 0.5s ease-in forwards;
    }
    
    .loader-text {
        font-size: 2rem;
        font-weight: 600;
        color: #F8F8F8; /* Milk White */
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        opacity: 0;
        animation: textFadeIn 1s ease-in-out forwards;
        animation-delay: 0.2s;
    }

    .loader-spinner {
        border: 4px solid rgba(248, 248, 248, 0.3); /* Milk White transparent */
        border-top: 4px solid #F8F8F8; /* Milk White solid */
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin-right: 15px;
        opacity: 0;
        animation: fadeIn 0.5s ease-out forwards;
        animation-delay: 0.1s;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; display: none; }
    }

    @keyframes textFadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Animations for chat messages */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .typing-indicator {
        animation: pulse 1.5s infinite;
    }
    
    /* Welcome Message Styling */
    .welcome-message {
        background: #F8F8F8; /* Milk White */
        border-radius: 18px;
        padding: 24px;
        margin: 20px auto;
        max-width: 80%;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideIn 0.8s ease-out;
        color: #1A202C; /* Dark Blue text */
    }
    
    .welcome-title {
        color: #1A202C; /* Dark Blue */
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    .welcome-text {
        color: #4A5568; /* Slightly lighter dark blue */
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 16px;
    }
    
    .quick-questions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: 16px;
    }
    
    .quick-question-btn {
        background: rgba(42, 59, 76, 0.1); /* Transparent dark blue */
        border: 1px solid rgba(42, 59, 76, 0.3);
        color: #1A202C; /* Dark Blue */
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .quick-question-btn:hover {
        background: rgba(42, 59, 76, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .chat-container {
            padding: 10px;
        }
        
        .stChatMessage {
            max-width: 95% !important;
            font-size: 14px !important;
        }
        
        .welcome-message {
            max-width: 95%;
            padding: 20px;
        }
        
        .welcome-title {
            font-size: 1.5rem;
        }
        
        .welcome-text {
            font-size: 1rem;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
    <div class="main-header">
        <h1 class="header-title">SIMATS Assistant</h1>
        <p class="header-subtitle">Your intelligent guide to Saveetha Institute of Medical and Technical Sciences</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.first_visit = True

# --- Custom Loading Screen ---
# This div will be shown initially and then hidden via JavaScript
st.markdown("""
    <div id="custom-loader-overlay">
        <div class="loader-spinner"></div>
        <div class="loader-text">Loading SIMATS data...</div>
    </div>
""", unsafe_allow_html=True)

# Load documents with custom loader message
retriever = load_and_process_documents()

# JavaScript to hide the loading overlay after data is loaded
st.markdown("""
    <script>
        const loader = document.getElementById('custom-loader-overlay');
        if (loader) {
            loader.classList.add('fade-out');
            setTimeout(() => {
                loader.style.display = 'none';
            }, 500); // Match fadeOut animation duration
        }
    </script>
""", unsafe_allow_html=True)


# Display welcome message only on first visit and add to chat history
if st.session_state.first_visit:
    st.markdown("""
        <div class="welcome-message">
            <h2 class="welcome-title">👋 Welcome to SIMATS!</h2>
            <p class="welcome-text">
                I'm here to help you discover everything about Saveetha Institute of Medical and Technical Sciences. 
                From admissions and courses to campus life and placements - just ask me anything!
            </p>
            <div class="quick-questions">
                <span class="quick-question-btn">What courses are offered?</span>
                <span class="quick-question-btn">Tell me about campus facilities</span>
                <span class="quick-question-btn">Placement opportunities</span>
                <span class="quick-question-btn">Admission process</span>
                <span class="quick-question-btn">College rankings</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    welcome_msg_content = """Hi there! I'm your SIMATS Assistant, and I'm excited to help you explore everything about our college! Whether you're curious about:

* **Academic programs** and specializations
* **Campus facilities** and infrastructure  
* **Placement opportunities** and career support
* **Admission requirements** and procedures
* **Rankings and accreditations**
* **Research opportunities** and projects
* **Hostel life** and campus culture

Just ask me anything! I'm here to give you detailed, friendly answers about SIMATS. What would you like to know first?"""
    
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg_content})
    st.session_state.first_visit = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about SIMATS..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        # We removed the st.spinner and replaced it with a custom CSS loading effect
        response_placeholder = st.empty() # Placeholder for typewriter effect
        response_placeholder.markdown("<div class='typing-indicator'>Thinking...</div>", unsafe_allow_html=True) # Show thinking indicator
        
        response = get_college_info_sync(prompt, retriever)
        
        # Add typing effect
        typewriter_effect(response, response_placeholder)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
