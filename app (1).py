import os
import google.generativeai as genai
import warnings
from fpdf import FPDF
import gradio as gr
import re
import unidecode
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore")

# Set your Google API key as an environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API')
if GOOGLE_API_KEY is None:
    raise ValueError("Google API key is not set. Make sure it's defined in your Hugging Face secret variables.")
# Configure the Google Generative AI library
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize model and embeddings
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2, convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Function to detect mathematical expressions in the answer
def contains_math(text):
    # Simple regex pattern to detect mathematical symbols or LaTeX expressions
    math_pattern = r'(\$.*?\$|\d+[\+\-\*/^]\d+|\\frac|\\sum|\\int|\\lim|\\sqrt)'
    return re.search(math_pattern, text) is not None

# Define function to process PDF and generate answers using standard model
def process_pdf_and_qa(pdf_file, question):
    # Load and split PDF
    pdf_loader = PyPDFLoader(pdf_file)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":6})

    # Standard template for question answering
    template = """
    Use the following pieces of context to answer the question at the end.
    If the answer to the question includes equations, display the equations using LaTeX format enclosed by `$$`.
    For inline equations, use single `$`, and for displayed equations, use double `$$`.
    If relevant, explain the meaning of the equation and any key terminology.
    If the question is beyond the context, generate an answer using general knowledge and mention that.
    Always say "thanks for asking!" at the end.
    Context:
    {context}
    Question: {question}
    Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Create the Retrieval QA chain using the standard model
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,  # Use the standard model for both cases
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Get the initial answer from the standard model
    result = qa_chain({"query": question})
    answer = result["result"]

    # Format answer for LaTeX if it contains math
    if contains_math(answer):
        answer = answer.replace('$', '\\( ').replace('$$', '\\[ ').replace('\\)', ' \\)').replace('\\[', ' \\]').replace('\\]', ' \\]')
    
    # Return the final answer
    return answer

# Define function to create PDF
def create_pdf(question, answer):
    # Replace unsupported characters with ASCII equivalents
    question = unidecode.unidecode(question)  # Convert to closest ASCII representation
    answer = unidecode.unidecode(answer)      # Handle non-latin1 characters
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Question:", ln=True)
    pdf.multi_cell(0, 10, txt=question)
    pdf.cell(200, 10, txt="Answer:", ln=True)
    pdf.multi_cell(0, 10, txt=answer)
    pdf_output = "/tmp/answer.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Define Gradio interface
def gradio_interface(pdf_file, question):
    answer = process_pdf_and_qa(pdf_file, question)
    pdf_path = create_pdf(question, answer)
    return answer, pdf_path

# Custom CSS for styling
custom_css = """
    .container {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .title {
        color: #2c3e50;
        font-size: 24px;
        font-weight: bold;
    }
    .description {
        color: #34495e;
        font-size: 16px;
        margin-bottom: 20px;
    }
    #question-input {
        border: 2px solid #3498db;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        color: #000;
    }
    .output-box {
        border: 2px solid #2ecc71;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        background-color: #ecf0f1;
    }
    .footer {
        font-size: 14px;
        color: #95a5a6;
        margin-top: 20px;
        text-align: center;
    }
"""

# Define the Gradio interface with styling
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.File(label="Upload PDF", elem_id="pdf-upload"), gr.Textbox(label="Ask a Question", elem_id="question-input")],
    outputs=[gr.Markdown(label="Answer", elem_id="answer-output"), gr.File(label="Download Answer as PDF", elem_id="pdf-download")],
    title="INTELLI_PDF",
    description="Upload a PDF file and ask questions related to its content. Download the answer as a PDF.",
    css=custom_css
)

if __name__ == "__main__":
    iface.launch(share=True)
