import os
import requests
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from query_processor import QueryProcessor
from PIL import Image
from io import BytesIO
import google.generativeai as genai

# Configuration functions
load_dotenv()
def configure_gemini():
    """Configure and return Gemini model"""

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
    genai.configure(api_key=google_api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def initialize_pinecone():
    """Initialize and return Pinecone index connection"""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index("o-level-physics-paper-1")

# Helper functions
def display_image(image_url):
    """Handle image display for both regular URLs and Google Drive links"""
    try:
        if 'drive.google.com' in image_url:
            file_id = image_url.split('/d/')[1].split('/')[0]
            direct_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            session = requests.Session()
            response = session.get(direct_url, stream=True)
            
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = image.resize((1000, 500))
                st.image(image, caption="Question Diagram", use_container_width=True)
            else:
                st.error(f"Failed to load Google Drive image (HTTP {response.status_code})")
        else:
            st.image(image_url, caption="Question Diagram", use_container_width=True)
            
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def create_generation_prompt(results):
    """Create prompt for question generation based on search results"""
    examples = []
    for match in results[:3]:
        meta = match.metadata
        example = f"Question: {meta['questionStatement']}\n"
        if meta.get('options') and 'https' not in meta.get('options')[0]:
            example += "Options:\n" + "\n".join(meta['options']) + "\n"
        examples.append(example)
    
    return f"""You are an expert O level examiner. Create new exam questions similar to these examples:

{'\n\n'.join(examples)}

Guidelines:
1. Generate 2-3 new questions
2. Follow O-Level standards
3. No images or links
4. Include multiple-choice options when applicable
6. Make questions original but similar in style to examples
7. Keep questions clear and self-contained"""

# Streamlit app configuration
st.set_page_config(
    page_title="WISSEN",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .question-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background: white;
        border-bottom: 1px solid #e0e0e0;
    }
    .header {color: #2b5876;}
    .generated-content {
        background: #fff7e6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.markdown("<h1 class='header'>📚 WISSEN</h1>", unsafe_allow_html=True)

# Initialize services
try:
    index = initialize_pinecone()
    query_processor = QueryProcessor(index)
    gemini_model = configure_gemini()
except Exception as e:
    st.error(f"Failed to initialize services: {str(e)}")
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.markdown("## About")
    st.markdown("Search through O-Level Physics past paper questions using natural language queries.")
    st.markdown("**Features:**")
    st.markdown("- Search by year, topic, or question content")
    st.markdown("- Generate new questions using AI")
    st.markdown("- View full question details")
    st.markdown("---")
    st.markdown("**Application Modes**")
    app_mode = st.radio(
        "Select Mode:",
        ["🔍 Retrieval Mode", "✨ Generation Mode"],
        index=0,
        help="Switch between finding existing questions or generating new ones"
    )
    st.markdown("---")
    st.markdown("**Example searches:**")
    st.markdown("- 'Questions about nuclear physics'")
    st.markdown("- '2023 magnetism problems'")
    st.markdown("- 'Mirror diagram questions'")

# Search form
with st.form("search_form"):
    query = st.text_input("Enter your question search:", 
                         placeholder="E.g. 'Find year 2023 questions about magnetism'")
    search_button = st.form_submit_button("🔍 Search")

# Handle search
if search_button and query:
    with st.spinner("Searching through physics questions..."):
        try:
            results = query_processor.search_questions(query)
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            st.stop()
    
    if not results:
        st.warning("No matching questions found. Try different keywords.")
    else:
        if "Generation" in app_mode:
            try:
                with st.spinner("Generating new questions using AI..."):
                    prompt = create_generation_prompt(results)
                    response = gemini_model.generate_content(prompt)
                    
                    if response.text:
                        st.markdown("## 🚀 Generated Questions")
                        # st.markdown("""
                        #     <div class="generated-content">
                        #     *AI-generated based on similar past questions*  
                        #     *Note: Always verify generated content for accuracy*
                        #     </div>
                        #     """, unsafe_allow_html=True)
                        st.divider()
                        st.markdown(response.text)
                    else:
                        st.error("Failed to generate questions. Please try again.")
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
        else:
            st.markdown(f"### Found {len(results)} results for: '{query}'")
            
            for i, match in enumerate(results, 1):
                meta = match.metadata
                with st.container():
                    st.divider()
                    
                    cols = st.columns([1,4])
                    with cols[0]:
                        st.markdown(f"**Year: ** {meta['year']}")
                        st.markdown(f"**Month: **{meta['months']}")
                        st.markdown(f"**Variant: ** {meta['variant']}")
                        st.markdown(f"**Question #: ** {meta['questionNumber']}")
                    
                    with cols[1]:
                        st.markdown(f"**Statement:** {meta['questionStatement']}")
                        
                        if meta.get('image'):
                            if isinstance(meta['image'],str) and meta['image'] not in ['', 'urlOfImage']:
                                display_image(meta['image'])
                            
                            if isinstance(meta['options'], list):
                                st.markdown("**Diagrams:**")
                                for img in meta['image']:
                                    if "https" in img:
                                        display_image(img)
                            
                        if meta.get('options') and isinstance(meta['options'], list):
                            st.markdown("**Options:**")
                            for opt in meta['options']:
                                if "https" in opt:
                                    display_image(opt)
                                else:
                                    st.markdown(f"- {opt}")

elif search_button and not query:
    st.error("Please enter a search query first!")