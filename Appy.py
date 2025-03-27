import streamlit as st
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
import google.generativeai as genai
import json
import io
import logging
import re

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Advisory Structure ---
DEPARTMENTS = [
    "Administration", "Finance & Accounts", "Training", "Procurement & Commercial",
    "Human Resource", "Information Security (InfoSec)", "Information Technology",
    "Internal Audit & Risk Management", "Business Operations", "Legal",
    "Marketing & Communication", "Mergers & Acquisitions", "Pre Sales & Sales",
    "Quality Assurance"
]
SUB_DEPARTMENTS = [
    "Operations - Cash Management", "Operations - Trade Services", "Operations - Custody operations",
    "Operations - Money Market", "Operations - FX", "Operations - ALM", "Operations - Retail Banking",
    "Operations - Debit/ Credit cards", "Market Risk", "Liquidity Risk", "Credit Risk",
    "Operational Risk", "Financial Risk", "Bankwide"
]
REGULATORY_RISK_THEME = [
    "Governance (Prudential and Licensing)", "Information and Technology", "Market Conduct and Customers",
    "Financial Crime", "Employment Practices and Workplace Safety", "Process Enhancement",
    "Conflict of Interest", "Risk"
]
RISK_CATEGORY_ID = [
    "Credit & Financial", "Non Implementation Risk", "Credit Regulatory Risk",
    "Reporting Risk", "Conflict of Interest", "Non Compliance Risk", "IT Security Risk"
]
INHERENT_RISK_RATING = ["Very High", "High", "Medium", "Low"]
NATURE_OF_MITIGATION = ["Policy", "Governance", "Process/controls", "Systems and Technology", "Training"]
COMPLIANCE_FREQUENCY = ["Onetime", "Daily", "Weekly", "Monthly", "Yearly"]
W_DAYOFWEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
M_WEEKSOFOCCURRENCEFORMONTHLY = ["First", "Second", "Third", "Fourth", "Last", "Day"]

# --- Helper Functions ---

def fetch_content(url):
    """Fetches content from the URL, handling potential errors."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=30, stream=True) # Use stream=True for PDFs
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        logging.error(f"Error fetching URL {url}: {e}")
        return None

def extract_html_text(response):
    """Extracts text content from HTML response using BeautifulSoup."""
    try:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Attempt to remove common boilerplate (header, footer, nav) - this is heuristic
        for tag in soup(['header', 'footer', 'nav', 'script', 'style', 'aside']):
            tag.decompose()
        # Get remaining text, joining paragraphs/sections
        paragraphs = [p.get_text(strip=True) for p in soup.find_all(['p', 'div', 'article', 'section'])]
        text = ' '.join(paragraphs)
        if not text: # Fallback if specific tags yield nothing
             text = soup.get_text(separator=' ', strip=True)
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully extracted text from HTML (approx {len(text)} chars).")
        return text, paragraphs if text else (None, None)
    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        logging.error(f"Error parsing HTML: {e}")
        return None, None

def extract_pdf_text(response):
    """Extracts text content from PDF response using pdfminer.six."""
    try:
        # Read PDF content into a BytesIO object
        pdf_content = io.BytesIO(response.content)
        text = extract_text(pdf_content)
         # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully extracted text from PDF (approx {len(text)} chars).")
        return text if text else None
    except Exception as e:
        # pdfminer can raise various errors, catch broadly
        st.error(f"Error extracting text from PDF: {e}. The PDF might be image-based or corrupted.")
        logging.error(f"Error extracting text from PDF: {e}")
        return None

def configure_gemini():
    """Configures the Gemini API."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        logging.info("Gemini API configured successfully.")
        return True
    except KeyError:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
        logging.error("GEMINI_API_KEY not found in Streamlit secrets.")
        return False
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        logging.error(f"Error configuring Gemini API: {e}")
        return False

def generate_compliance_advisory(text, paragraphs):
    """Generates structured compliance advisory using Gemini LLM."""
    if not configure_gemini():
        return None

    model = genai.GenerativeModel('gemini-1.5-flash') # Or use 'gemini-pro' if preferred

    # --- Detailed Prompt Construction ---
    prompt = f"""
    Analyze the following text extracted from an RBI notification and generate a SINGLE structured compliance advisory in JSON format.
    Adhere strictly to the specified JSON structure and field constraints. Select only from the provided options where applicable.
    Base your answers *solely* on the provided text. If information isn't present, make the most reasonable inference or use the specified defaults.

    **RBI Notification Text:**
    ```
    {text[:15000]}
    ```
    *(Note: Text might be truncated for brevity)*

    **Required JSON Output Format:**

    ```json
    {{
      "Departments": "Select ONE from: {', '.join(DEPARTMENTS)}",
      "Task Name": "Start with 'To...'. Brief action statement.",
      "Task Description": "Clear, concise description of the required action, risk, or procedure in simple language.",
      "Sub-Departments": "Select ONE from: {', '.join(SUB_DEPARTMENTS)}",
      "Regulatory Risk/Theme": "Select ONE from: {', '.join(REGULATORY_RISK_THEME)}",
      "Risk Category ID": "Select ONE from: {', '.join(RISK_CATEGORY_ID)}",
      "Inherent Risk Rating": "Select ONE from: {', '.join(INHERENT_RISK_RATING)}",
      "Nature of Mitigation": "Select ONE from: {', '.join(NATURE_OF_MITIGATION)}",
      "Task Frequency Start Date": "Use '01-04-2024' for recurring tasks (Daily, Weekly, Monthly, Yearly). Use the notification's effective/issue date for 'Onetime' tasks if identifiable, otherwise default to '01-04-2024'. Format: DD-MM-YYYY",
      "Task Frequency End Date": "Use '31-03-2039'. Format: DD-MM-YYYY",
      "Compliance Frequency": "Select ONE from: {', '.join(COMPLIANCE_FREQUENCY)}",
      "Due Date": "Use '31-03-2099' for 'Onetime' tasks. For recurring tasks, estimate the *first* due date based on frequency and start date if possible, otherwise use '31-03-2099'. Format: DD-MM-YYYY",
      "D_daysofoccurrence": "Specify interval (e.g., 1) ONLY if Compliance Frequency is 'Daily'. Otherwise, null or omit.",
      "W_weeksofoccurrence": "Specify interval (e.g., 1) ONLY if Compliance Frequency is 'Weekly'. Otherwise, null or omit.",
      "W_dayofweek": "Select ONE from: {', '.join(W_DAYOFWEEK)} ONLY if Compliance Frequency is 'Weekly'. Otherwise, null or omit.",
      "M_weeksofoccurrenceformonthly": "Select ONE from: {', '.join(M_WEEKSOFOCCURRENCEFORMONTHLY)} ONLY if Compliance Frequency is 'Monthly'. Otherwise, null or omit.",
      "M_dayofweek": "Specify day (e.g., 15) ONLY if Compliance Frequency is 'Monthly'. Otherwise, null or omit.",
      "M_monthsofoccurence": "Specify interval (e.g., 1) ONLY if Compliance Frequency is 'Monthly'. Otherwise, null or omit.",
      "Y_monthofyear": "Specify month (e.g., 'April') ONLY if Compliance Frequency is 'Yearly'. Otherwise, null or omit.",
      "Y_dayofmonth": "Specify day (e.g., 15) ONLY if Compliance Frequency is 'Yearly'. Otherwise, null or omit.",
      "References": "List of paragraph numbers (1-indexed) that were used to derive the information."
    }}
    ```

    **Instructions:**
    1. Carefully read the text to understand the core compliance requirement(s). Focus on identifying the main actionable task.
    2. Fill *every* field in the JSON structure.
    3. Strictly use the provided options for categorical fields. Do not invent new categories.
    4. Make logical choices for dates and frequencies based on the text. Apply defaults as specified.
    5. Ensure the final output is a single, valid JSON object.
    6. Include references to the paragraphs that were used to derive the information.
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                # Ensure JSON output if the model supports it directly
                 response_mime_type="application/json",
                temperature=0.2 # Lower temperature for more deterministic output
            )
        )
        # Clean potential markdown/code blocks if necessary
        generated_text = response.text.strip()
        if generated_text.startswith("```json"):
            generated_text = generated_text[7:]
        if generated_text.endswith("```"):
            generated_text = generated_text[:-3]

        logging.info("Received response from LLM.")
        # Validate and parse the JSON
        advisory_json = json.loads(generated_text)
        logging.info("Successfully parsed LLM response as JSON.")
        return advisory_json

    except json.JSONDecodeError as e:
        st.error(f"LLM response was not valid JSON: {e}")
        st.text_area("Raw LLM Response:", generated_text, height=200)
        logging.error(f"LLM response JSON parsing failed: {e}. Response: {generated_text}")
        return None
    except Exception as e:
        st.error(f"Error during LLM generation: {e}")
        logging.error(f"Error during LLM generation: {e}")
        # You might want to inspect `response.prompt_feedback` here for specific issues like blocked prompts
        # try:
        #    st.warning(f"Prompt Feedback: {response.prompt_feedback}")
        #    logging.warning(f"Prompt Feedback: {response.prompt_feedback}")
        # except Exception:
        #    pass # Ignore if feedback isn't available
        return None

# --- Streamlit App ---

st.set_page_config(page_title="RBI Compliance Advisory Generator", layout="wide")
st.title("🤖 RBI Compliance Advisory Generator")
st.markdown("""
Paste the URL of an RBI notification (HTML page or direct PDF link) below.
The application will attempt to extract the text, analyze it using an AI model,
and generate a structured compliance advisory based on the predefined format.

**Disclaimer:** This is an AI-powered tool. Always verify the generated advisory against the original RBI notification.
""")

url = st.text_input("Enter RBI Notification URL:", placeholder="https://www.rbi.org.in/...")

if st.button("Generate Advisory"):
    if not url:
        st.warning("Please enter a URL.")
    elif not (url.startswith("http://") or url.startswith("https://")):
         st.error("Invalid URL. Please enter a valid HTTP or HTTPS URL.")
    else:
        with st.spinner("Processing... Fetching content, extracting text, and analyzing with AI..."):
            # 1. Fetch Content
            response = fetch_content(url)
            extracted_text, paragraphs = None, None

            if response:
                content_type = response.headers.get('Content-Type', '').lower()
                is_pdf = 'application/pdf' in content_type or url.lower().endswith('.pdf')
                is_html = 'text/html' in content_type

                # 2. Extract Text
                if is_pdf:
                    logging.info(f"Detected PDF content type for URL: {url}")
                    extracted_text = extract_pdf_text(response)
                elif is_html:
                    logging.info(f"Detected HTML content type for URL: {url}")
                    extracted_text, paragraphs = extract_html_text(response)
                else:
                    # Fallback attempt for unknown content type - try HTML first
                    logging.warning(f"Unknown content type '{content_type}'. Attempting HTML extraction for URL: {url}")
                    extracted_text, paragraphs = extract_html_text(response)
                    if not extracted_text:
                         logging.warning(f"HTML extraction failed for unknown type. URL: {url}")
                         st.warning(f"Could not determine content type ('{content_type}'). If this is a PDF, ensure the URL ends with '.pdf'.")

                # 3. Generate Advisory (if text extracted)
                if extracted_text:
                    if len(extracted_text) < 100: # Basic check for meaningful content
                         st.warning("Extracted text seems very short. The notification might be empty, image-based, or complex. Results may be inaccurate.")
                         logging.warning(f"Extracted text is very short ({len(extracted_text)} chars) for URL: {url}")

                    advisory = generate_compliance_advisory(extracted_text, paragraphs)

                    # 4. Display Output
                    if advisory:
                        st.success("✅ Compliance Advisory Generated Successfully!")
                        st.json(advisory) # Display as interactive JSON

                        # Optionally display as formatted text/table
                        st.subheader("Formatted Advisory Details:")
                        # Use columns for better layout
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Department:** {advisory.get('Departments', 'N/A')}")
                            st.markdown(f"**Sub-Department:** {advisory.get('Sub-Departments', 'N/A')}")
                            st.markdown(f"**Task Name:** {advisory.get('Task Name', 'N/A')}")
                            st.markdown(f"**Task Description:**")
                            st.markdown(f"> {advisory.get('Task Description', 'N/A')}") # Blockquote for description
                            st.markdown(f"**Regulatory Risk/Theme:** {advisory.get('Regulatory Risk/Theme', 'N/A')}")
                            st.markdown(f"**Risk Category ID:** {advisory.get('Risk Category ID', 'N/A')}")
                            st.markdown(f"**Inherent Risk Rating:** {advisory.get('Inherent Risk Rating', 'N/A')}")
                            st.markdown(f"**Nature of Mitigation:** {advisory.get('Nature of Mitigation', 'N/A')}")
                        with col2:
                            st.markdown(f"**Compliance Frequency:** {advisory.get('Compliance Frequency', 'N/A')}")
                            st.markdown(f"**Task Frequency Start Date:** {advisory.get('Task Frequency Start Date', 'N/A')}")
                            st.markdown(f"**Task Frequency End Date:** {advisory.get('Task Frequency End Date', 'N/A')}")
                            st.markdown(f"**Due Date:** {advisory.get('Due Date', 'N/A')}")
                            # Display frequency details conditionally
                            freq = advisory.get('Compliance Frequency')
                            if freq == 'Daily':
                                st.markdown(f"**Daily Occurrence:** Every {advisory.get('D_daysofoccurrence', 'N/A')} day(s)")
                            elif freq == 'Weekly':
                                st.markdown(f"**Weekly Occurrence:** Every {advisory.get('W_weeksofoccurrence', 'N/A')} week(s) on {advisory.get('W_dayofweek', 'N/A')}")
                            elif freq == 'Monthly':
                                st.markdown(f"**Monthly Occurrence:** {advisory.get('M_weeksofoccurrenceformonthly', 'N/A')} {advisory.get('M_dayofweek', 'day')} every {advisory.get('M_monthsofoccurence', 'N/A')} month(s)")
                            elif freq == 'Yearly':
                                st.markdown(f"**Yearly Occurrence:** On {advisory.get('Y_monthofyear', 'N/A')} {advisory.get('Y_dayofmonth', 'N/A')}")

                        # Display paragraph references
                        st.subheader("References:")
                        references = advisory.get("References", [])
                        for ref in references:
                            st.markdown(f"**Paragraph {ref}:** {paragraphs[ref-1] if ref <= len(paragraphs) else 'Paragraph not found'}")

                    else:
                        st.error("💥 Failed to generate compliance advisory after extracting text.")
                else:
                    st.error("💥 Failed to extract text content from the URL. Cannot proceed.")
            else:
                # Error handled in fetch_content
                pass # Error message already shown by fetch_content

# --- Footer/Instructions ---
st.markdown("---")
st.markdown("Created with Streamlit and Google Gemini.")
