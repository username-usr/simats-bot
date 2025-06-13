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
GROQ_API_KEY = "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"  # Replace with your actual key
GOOGLE_API_KEY = "AIzaSyB5dlbtndihCliWB1GCXoZJaTwVYXidBVg"  # Replace with your actual key

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

# --- Streamlit App Setup ---
st.title("SIMATS Chatbot")
st.write("Your guide to Saveetha Institute of Medical and Technical Sciences")

# Load documents
retriever = load_and_process_documents()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.first_visit = True

# Display welcome message on first visit
if st.session_state.first_visit:
    welcome_msg = """Hi there! I'm your SIMATS Assistant, and I'm here to help you explore everything about Saveetha Institute of Medical and Technical Sciences. Ask about academic programs, campus facilities, placements, admissions, or anything else you're curious about! What would you like to know?"""
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    st.session_state.first_visit = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about SIMATS..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Fetching answer..."):
            response = get_college_info_sync(prompt, retriever)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
