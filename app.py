import sys
import argparse
import fitz  # PyMuPDF for PDF text extraction
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file given its file path.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    return full_text

def create_gap_analysis_prompt(isms_text, iso_excerpt):
    """
    Combines the ISO excerpt and the ISMS text into one prompt.
    """
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
    """
    Uses the Hugging Face Transformers pipeline with the free model "distilgpt2"
    to generate the gap analysis report.
    """
    generator = pipeline("text-generation", model="distilgpt2", device=-1)
    response = generator(prompt, max_length=1000, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

def main():
    parser = argparse.ArgumentParser(description="ISO 27001 Policy Gap Analysis Agent (CLI - Free LLM)")
    parser.add_argument("--pdf", required=True, help="Path to the ISMS policy PDF document")
    parser.add_argument("--iso_excerpt", required=True, help="Path to a text file containing the ISO 27001 excerpt")
    args = parser.parse_args()

    try:
        isms_text = extract_text_from_pdf(args.pdf)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        sys.exit(1)

    try:
        with open(args.iso_excerpt, "r", encoding="utf-8") as f:
            iso_excerpt_text = f.read()
    except Exception as e:
        print(f"Error reading ISO excerpt file: {e}")
        sys.exit(1)

    prompt = create_gap_analysis_prompt(isms_text, iso_excerpt_text)
    print("Generating policy gap analysis report using free LLM...")
    report = get_gap_analysis_report(prompt)
    output_file = "policy_gap_analysis_report.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Policy gap analysis report saved to {output_file}")

if __name__ == "__main__":
    main()
