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


GROQ_API_KEY = "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"
GOOGLE_API_KEY = "AIzaSyB5dlbtndihCliWB1GCXoZJaTwVYdXiBVg"

reader = easyocr.Reader(['en'])
chat_model = ChatGroq(api_key=GROQ_API_KEY, model="qwen-2.5-32b")


chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a friendly and knowledgeable assistant who provides clear and engaging information about SIMATS engineering college. You have access to official college documents.

**When responding, follow these guidelines:**
1. **Humanize your tone**: Write as if you’re explaining to a friend—use contractions ("you’ll", "we’ve"), occasional humor, and relatable examples.
   - *Example*: "Think of this like choosing between Netflix plans—but for courses!"
2. **Structure for clarity**:
   - Start with a **1-sentence summary**.
   - Use **bullet points** for key details.
   - Add a **real-world scenario** or analogy if helpful.
3. **Adapt to the user**:
   - If the question is formal (e.g., about deadlines), keep it professional.
   - If casual (e.g., “What’s campus life like?”), be conversational and don't tell i don't know or there is no information from the provided document.

**Avoid**:
- Robotic phrases like "Based on the provided context..."
- Overloading with jargon. Simplify technical terms.
"""),
    HumanMessagePromptTemplate.from_template("""**Context from college documents**:
{context}

**Question**: {question}

**Task**: Craft a response that:
1. Starts with a **hook** (e.g., “Great question!” or “Let’s break this down...”).
2. Answers *all* parts of the question with **blended context** (no copy-paste).
3. Uses **examples** (e.g., “For instance, last year a student...”).
4. Ends with:
""")
])

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
# Cache document loading and vector store creation
@st.cache_resource
def load_and_process_documents():
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
    "https://simatsengineering.com/profile",
    "https://simatsengineering.com/infrastructure",
    "https://simatsengineering.com/best-practices",
    "https://drive.google.com/file/d/1u90Awzw6iLUs-pxQdjUB9jyf2taojX-j/view",
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
    "https://sites.google.com/saveetha.com/sseappliedmachinelearning/home?pli=1",
    "https://sites.google.com/saveetha.com/department-of-physical-science/home",
    "https://sites.google.com/saveetha.com/simatssc/home",
    "https://simatsengineering.com/ai-ds",
    "https://simatsengineering.com/admissions",
    "https://simatsengineering.com/placement",
    "https://simatsengineering.com/iic",
    "https://simatsengineering.com/news-1",
    "https://simatsengineering.com/contact-us",
    "https://sites.google.com/saveetha.com/dmc/research?authuser=0",
    "https://scholar.google.co.in/citations?user=mcDa3D4AAAAJ&hl=en",
    "https://scholar.google.co.jp/citations?user=AQszbw4AAAAJ&hl=en",
    "https://scholar.google.co.jp/citations?user=VtgK7_oAAAAJ&hl=en",
    "https://sites.google.com/saveetha.com/dmc/research-scholar?authuser=0",
    "https://sites.google.com/saveetha.com/dmc/publications?authuser=0",
    "https://sites.google.com/saveetha.com/dmc/patents?authuser=0",
    "https://sites.google.com/saveetha.com/dmc/events?authuser=0",
    "https://sites.google.com/saveetha.com/dmc/contact-us?authuser=0",
    "https://sites.google.com/saveetha.com/dmc/home?authuser=0",
    "https://www.collegebatch.com/12013-saveetha-school-of-engineering-faculty-profiles-chennai",
    "https://www.collegebatch.com/12013-saveetha-school-of-engineering-fees-structure-courses-chennai",
    "https://www.collegebatch.com/12013-saveetha-school-of-engineering-placements-chennai",
    "https://www.collegebatch.com/12013-saveetha-school-of-engineering-hostel-fees-facilities-chennai",
    "https://www.collegebatch.com/12013-saveetha-school-of-engineering-contact-number-address-map-chennai",
    "https://www.collegebatch.com/12013-saveetha-school-of-engineering-reviews",
    "https://www.collegebatch.com/12013-saveetha-school-of-engineering-in-chennai",
    "https://www.knowafest.com/explore/college/Saveetha_Institute_of_Medical_and_Technical_Sciences_Engineering",
    "https://www.knowafest.com/explore/events/2025/03/0721-biociencia-2k25-saveetha-institute-medical-technical-sciences-engineering-national-conference-chennai",
    "https://www.knowafest.com/explore/events/2025/02/2709-minnal-25-saveetha-institute-medical-technical-sciences-engineering-national-level-symposium-chennai",
    "https://www.knowafest.com/explore/events/2025/02/1513-international-conference-neural-evolution-adaptive-intelligence-2025-saveetha-institute-medical-technical-sciences-engineering-chennai",
    "https://www.knowafest.com/explore/events/2025/01/2609-cricket-clash-smash-twenty-challenge-2025-saveetha-institute-medical-technical-sciences-engineering-inter-intra-college-competition-chennai",
    "https://simatsengineering.com/collaborations",
    "https://simatsengineering.com/news",
    "https://mmtechitservices.net/publications-in-database/",
    "https://mmtechitservices.net/research-lab/",
    "https://mmtechitservices.net/placement-training/",
    "https://mmtechitservices.net/campus-placements/",
    "https://mmtechitservices.net/internship/",
    "https://mmtechitservices.net/international-visiting-faculties/",
    "https://mmtechitservices.net/international-mou/",
    "https://mmtechitservices.net/stepup-saveetha-innovation-incubation-cell/",
    "https://mmtechitservices.net/companies-under-incubation/",
    "https://mmtechitservices.net/our-profile/",
    "https://mmtechitservices.net/our-services/",
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

# Helper functions
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

# Synchronous wrapper for async function
def get_college_info_sync(question, retriever):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(get_college_info(question, retriever))
    loop.close()
    return result

# Minimal CSS with WhatsApp-like design and wider container
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400&display=swap');
    html, body, [data-testid="stApp"] {
        background-color: #202123 !important;
        color: #D9D9D9 !important;
        font-family: 'Roboto', sans-serif !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stAppViewContainer"] {
        max-width: 100vw !important;  /* Increased from 800px to 1200px to reduce empty space */
        margin: 0 auto !important;    /* Center the container */
        height: 100vh !important;
        background-color: #202123 !important;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5) !important;
    }
    h1 {
        background-color: #10A37F !important;
        color: #D9D9D9 !important;
        padding: 10px 15px !important;
        margin: 0 !important;
        font-size: 30px !important;
        text-align: left !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    .stChatMessage {
        padding: 10px 15px !important;
        margin: 5px 5px !important;
        font-size: 24px !important;
        border-radius: 10px !important;
        max-width: 95% !important;
        word-wrap: break-word !important;
    }
    .stChatMessage.user {
        background-color: #343541 !important;
        margin-left: auto !important;
    }
    .stChatMessage.assistant {
        background-color: #444654 !important;
        margin-right: auto !important;
    }
    .stTextInput > div > input {
        background-color: #343541 !important;
        color: #D9D9D9 !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 10px 15px !important;
        font-size: 28px !important;
        width: calc(100% - 20px) !important;
        margin: 10px 10px !important;
    }
    .stSpinner > div {
        border-color: #10A37F transparent transparent transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main UI
retriever = load_and_process_documents()

st.title("SIMATS Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about SIMATS..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            response = get_college_info_sync(prompt, retriever)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})