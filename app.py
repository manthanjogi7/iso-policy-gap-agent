import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from transformers import pipeline

def extract_text_from_pdf_stream(file_obj):
    """Extract text from a PDF file provided via Streamlit uploader."""
    doc = fitz.open("pdf", file_obj.read())
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text

def create_gap_analysis_prompt(isms_text, iso_excerpt):
    """Combine the ISO excerpt and the ISMS policy text into a prompt."""
    prompt = f"""
You are an ISO 27001 compliance expert. The following is an excerpt from the ISO 27001 standard:
{iso_excerpt}

The following is the text from the organization's ISMS policy document:
{isms_text}

Please analyze the gap between the organization's policy and the ISO 27001 standard. Provide a detailed policy gap analysis report that:
1. Identifies areas where the policy does not meet the ISO 27001 requirements.
2. Provides clear recommendations for policy improvements, including alternative policy language.

Output the report in a clear, structured format.
"""
    return prompt

def get_gap_analysis_report(prompt):
    """Uses the Hugging Face Transformers pipeline with a free chat model to generate the gap analysis report."""
    generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device=-1)
    response = generator(prompt, max_length=1000, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

def main():
    st.title("ISO 27001 Policy Gap Analysis Agent (Free LLM)")
    st.write("Upload your ISMS policy PDF, paste the ISO 27001 excerpt, and click the button to generate the gap analysis report using a free, open-source LLM.")

    uploaded_pdf = st.file_uploader("Upload ISMS PDF", type=["pdf"])
    iso_excerpt = st.text_area("ISO 27001 Excerpt", value="[Paste your ISO 27001 excerpt here]")
    # API key is optional in this free LLM setup; you can remove the field if desired.
    api_key = st.text_input("(Optional) API Key", type="password")

    if st.button("Run Gap Analysis"):
        if not uploaded_pdf:
            st.error("Please upload a PDF file.")
        elif not iso_excerpt.strip():
            st.error("Please enter the ISO 27001 excerpt.")
        else:
            st.info("Extracting text from PDF...")
            isms_text = extract_text_from_pdf_stream(uploaded_pdf)
            prompt = create_gap_analysis_prompt(isms_text, iso_excerpt)
            st.info("Generating gap analysis report using free LLM...")
            report = get_gap_analysis_report(prompt)
            st.subheader("Policy Gap Analysis Report")
            st.text_area("Report Output", value=report, height=400)
            st.download_button(
                label="Download Report",
                data=report,
                file_name="policy_gap_analysis_report.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
