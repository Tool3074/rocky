import streamlit as st
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
import google.generativeai as genai
from groq import Groq, RateLimitError as GroqRateLimitError, APIError as GroqAPIError
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.exceptions import MistralAPIException, MistralConnectionException
from openai import OpenAI, RateLimitError as OpenAIRateLimitError, APIError as OpenAIAPIError # For OpenRouter
import json
import io
import logging
import time
import re
import os # For potential env var usage in OpenRouter headers

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MAX_RETRIES = 1 # Number of retries after a rate limit error
COOLING_PERIOD_SECONDS = 60 # Delay in seconds before retrying

# --- Constants for Advisory Structure (Keep these as before) ---
DEPARTMENTS = [
    "Administration", "Finance & Accounts", "Training", "Procurement & Commercial",
    "Human Resource", "Information Security (InfoSec)", "Information Technology",
    "Internal Audit & Risk Management", "Business Operations", "Legal",
    "Marketing & Communication", "Mergers & Acquisitions", "Pre Sales & Sales",
    "Quality Assurance"
]
# ... (rest of the constants: SUB_DEPARTMENTS, REGULATORY_RISK_THEME, etc. remain unchanged)
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

# --- Provider Choices ---
LLM_PROVIDERS = ["Gemini", "Groq", "Mistral", "OpenRouter"]
# Define default/recommended models for each provider
DEFAULT_MODELS = {
    "Gemini": "gemini-1.5-flash",
    "Groq": "llama3-8b-8192", # Fast option on Groq
    "Mistral": "mistral-small-latest", # Balanced option on Mistral
    "OpenRouter": "google/gemini-flash-1.5" # Example: Use Gemini via OpenRouter
    # Add more OpenRouter models if needed: e.g., "mistralai/mistral-7b-instruct"
}

# --- Helper Functions (fetch_content, extract_html_text, extract_pdf_text - remain unchanged) ---
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
        text = ' '.join(p.get_text(strip=True) for p in soup.find_all(['p', 'div', 'article', 'section']))
        if not text: # Fallback if specific tags yield nothing
             text = soup.get_text(separator=' ', strip=True)
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        logging.info(f"Successfully extracted text from HTML (approx {len(text)} chars).")
        return text if text else None
    except Exception as e:
        st.error(f"Error parsing HTML: {e}")
        logging.error(f"Error parsing HTML: {e}")
        return None

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

# --- LLM Prompt Generation ---
def create_llm_prompt(text):
    """Creates the detailed prompt for the LLM."""
    # Truncate text to avoid exceeding input token limits
    # Note: 15k chars is a rough estimate; actual token count varies by model
    truncated_text = text[:15000]
    if len(text) > 15000:
        logging.warning(f"Input text truncated to 15000 characters from original {len(text)}.")

    return f"""
    Analyze the following text extracted from an RBI notification and generate a SINGLE structured compliance advisory in JSON format.
    Adhere strictly to the specified JSON structure and field constraints. Select only from the provided options where applicable.
    Base your answers *solely* on the provided text. If information isn't present, make the most reasonable inference or use the specified defaults.

    **RBI Notification Text:**
    ```
    {truncated_text}
    ```

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
      "Y_dayofmonth": "Specify day (e.g., 15) ONLY if Compliance Frequency is 'Yearly'. Otherwise, null or omit."
    }}
    ```

    **Instructions:**
    1. Carefully read the text to understand the core compliance requirement(s). Focus on identifying the main actionable task.
    2. Fill *every* field in the JSON structure.
    3. Strictly use the provided options for categorical fields. Do not invent new categories.
    4. Make logical choices for dates and frequencies based on the text. Apply defaults as specified.
    5. Ensure the final output is a single, valid JSON object. Do not include any explanatory text before or after the JSON object itself.
    """

# --- LLM API Call Functions ---

def generate_with_gemini(prompt, api_key, model_name):
    """Generates advisory using Google Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        # Handle potential safety blocks or empty responses
        if not response.parts:
             raise Exception(f"Gemini response blocked or empty. Feedback: {response.prompt_feedback}")
        return response.text
    except Exception as e:
        # Specific error handling for Gemini can be added if needed
        logging.error(f"Gemini API Error: {e}")
        # Check for potential rate limit hints (though Gemini uses quota, not simple rate limits)
        if "quota" in str(e).lower():
             raise RateLimitExceededError(f"Gemini API quota likely exceeded: {e}") from e
        raise # Re-raise other errors

def generate_with_groq(prompt, api_key, model_name):
    """Generates advisory using Groq."""
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return chat_completion.choices[0].message.content
    except GroqRateLimitError as e:
        logging.error(f"Groq Rate Limit Error: {e}")
        raise RateLimitExceededError("Groq rate limit hit.") from e
    except GroqAPIError as e:
        logging.error(f"Groq API Error: {e}")
        raise # Re-raise other API errors
    except Exception as e:
        logging.error(f"Groq General Error: {e}")
        raise

def generate_with_mistral(prompt, api_key, model_name):
    """Generates advisory using Mistral AI."""
    try:
        client = MistralClient(api_key=api_key)
        chat_response = client.chat(
            model=model_name,
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return chat_response.choices[0].message.content
    except MistralAPIException as e:
        logging.error(f"Mistral API Error: Status Code: {e.status_code}, Message: {e.message}")
        if e.status_code == 429: # Explicit check for rate limit status code
             raise RateLimitExceededError("Mistral rate limit hit.") from e
        raise # Re-raise other API errors
    except MistralConnectionException as e:
         logging.error(f"Mistral Connection Error: {e}")
         raise # Re-raise connection errors
    except Exception as e:
        logging.error(f"Mistral General Error: {e}")
        raise

def generate_with_openrouter(prompt, api_key, model_name):
    """Generates advisory using OpenRouter (OpenAI compatible API)."""
    # Optional: Get referrer URL from secrets or environment variable
    referer = st.secrets.get("OPENROUTER_REFERRER", os.environ.get("OPENROUTER_REFERRER", ""))
    site_name = st.secrets.get("OPENROUTER_SITE_NAME", os.environ.get("OPENROUTER_SITE_NAME", "StreamlitApp"))

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
            # Add recommended headers for OpenRouter tracking
            extra_headers={
                "HTTP-Referer": referer,
                "X-Title": site_name,
            } if referer else None
        )
        return response.choices[0].message.content
    except OpenAIRateLimitError as e: # Catch OpenAI specific rate limit error
        logging.error(f"OpenRouter Rate Limit Error: {e}")
        raise RateLimitExceededError("OpenRouter rate limit hit.") from e
    except OpenAIAPIError as e:
         logging.error(f"OpenRouter API Error: Status Code: {e.status_code}, Message: {e.message}")
         if e.status_code == 429: # Check status code just in case
              raise RateLimitExceededError("OpenRouter rate limit hit (via status code).") from e
         # Handle input token limit errors specifically if possible (often 400 Bad Request)
         if e.status_code == 400 and "maximum context length" in str(e).lower():
              logging.error(f"OpenRouter Error: Input likely exceeds model's maximum context length.")
              st.error(f"Error: The extracted text is too long for the selected model ({model_name}). Try a different model or a shorter notification.")
              # Do not retry for this error, raise a specific exception or return None
              raise ValueError("Input text exceeds model context length.") from e
         raise # Re-raise other API errors
    except Exception as e:
        logging.error(f"OpenRouter General Error: {e}")
        raise


# Custom Exception for unified rate limit handling
class RateLimitExceededError(Exception):
    pass

# --- Main Generation Logic with Retry ---
def generate_compliance_advisory(text, provider, model_name):
    """
    Generates structured compliance advisory using the selected LLM provider.
    Includes retry logic for rate limit errors.
    """
    prompt = create_llm_prompt(text)
    api_key = None
    generate_func = None

    # --- Get API Key and Function based on provider ---
    try:
        if provider == "Gemini":
            api_key = st.secrets["GEMINI_API_KEY"]
            generate_func = generate_with_gemini
        elif provider == "Groq":
            api_key = st.secrets["GROQ_API_KEY"]
            generate_func = generate_with_groq
        elif provider == "Mistral":
            api_key = st.secrets["MISTRAL_API_KEY"]
            generate_func = generate_with_mistral
        elif provider == "OpenRouter":
            api_key = st.secrets["OPENROUTER_API_KEY"]
            generate_func = generate_with_openrouter
        else:
            st.error(f"Invalid provider selected: {provider}")
            return None

        if not api_key:
            st.error(f"API Key for {provider} not found in Streamlit secrets. Please add it.")
            logging.error(f"API Key for {provider} not found.")
            return None

    except KeyError as e:
        st.error(f"Missing API Key in Streamlit secrets: {e}. Please add it.")
        logging.error(f"Missing API Key in Streamlit secrets: {e}")
        return None
    except Exception as e:
        st.error(f"Error during configuration: {e}")
        logging.error(f"Error configuring for provider {provider}: {e}")
        return None

    # --- API Call with Retry Logic ---
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            start_time = time.time()
            logging.info(f"Attempting API call to {provider} (Attempt {retries + 1}/{MAX_RETRIES + 1})")
            raw_response = generate_func(prompt, api_key, model_name)
            end_time = time.time()
            logging.info(f"Received response from {provider} in {end_time - start_time:.2f} seconds.")

            # Clean potential markdown/code blocks (optional, as JSON mode should prevent this)
            generated_text = raw_response.strip()
            if generated_text.startswith("```json"):
                generated_text = generated_text[7:]
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3]

            # Validate and parse the JSON
            advisory_json = json.loads(generated_text)
            logging.info("Successfully parsed LLM response as JSON.")
            return advisory_json # Success

        except RateLimitExceededError as rlee:
            logging.warning(f"Rate limit encountered for {provider}: {rlee}. Attempt {retries + 1}/{MAX_RETRIES + 1}.")
            if retries < MAX_RETRIES:
                st.warning(f"Rate limit hit for {provider}. Waiting {COOLING_PERIOD_SECONDS} seconds before retrying...")
                time.sleep(COOLING_PERIOD_SECONDS)
                retries += 1
            else:
                st.error(f"Rate limit hit for {provider}, and max retries ({MAX_RETRIES}) exceeded. Please try again later.")
                logging.error(f"Rate limit hit for {provider}, max retries exceeded.")
                return None # Failed after retries

        except json.JSONDecodeError as e:
            st.error(f"LLM response from {provider} was not valid JSON: {e}")
            st.text_area("Raw LLM Response:", generated_text if 'generated_text' in locals() else 'N/A', height=200)
            logging.error(f"{provider} response JSON parsing failed: {e}. Response: {generated_text if 'generated_text' in locals() else 'N/A'}")
            return None # Failed - invalid JSON

        except ValueError as ve: # Catch specific value errors like context length exceeded
             if "context length" in str(ve):
                  # Error already logged and shown by the provider function
                  return None
             else:
                  # Handle other unexpected ValueErrors
                  st.error(f"An unexpected value error occurred with {provider}: {ve}")
                  logging.error(f"Unexpected ValueError with {provider}: {ve}")
                  return None


        except Exception as e:
            st.error(f"An unexpected error occurred during LLM generation with {provider}: {e}")
            logging.error(f"Error during LLM generation with {provider} (Attempt {retries + 1}): {e}")
            # Optional: Show more specific feedback if available (e.g., prompt feedback for Gemini)
            # try:
            #    if provider == "Gemini" and 'response' in locals():
            #       st.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")
            #       logging.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")
            # except Exception: pass
            # Decide if retry makes sense for generic errors (potentially not)
            # For simplicity here, we'll stop after the first generic error.
            return None # Failed - other error

    return None # Should not be reached if loop logic is correct

# --- Streamlit App ---

st.set_page_config(page_title="RBI Compliance Advisory Generator", layout="wide")
st.title("🤖 RBI Compliance Advisory Generator")
st.markdown("""
Paste the URL of an RBI notification (HTML page or direct PDF link) below.
Select the AI provider, and the application will attempt to extract the text, analyze it,
and generate a structured compliance advisory.

**Disclaimer:** This is an AI-powered tool. Always verify the generated advisory against the original RBI notification. Performance may vary between AI providers.
""")

# --- UI Elements ---
url = st.text_input("Enter RBI Notification URL:", placeholder="https://www.rbi.org.in/...")

# Provider Selection
selected_provider = st.selectbox("Choose AI Provider:", LLM_PROVIDERS)

# Dynamically set the default model based on the selected provider
default_model = DEFAULT_MODELS.get(selected_provider, "")
# You could enhance this with more model choices per provider if desired
# For now, we use the default. Ensure DEFAULT_MODELS has entries for all in LLM_PROVIDERS
st.caption(f"Using model: `{default_model}` for {selected_provider}")

if st.button(f"Generate Advisory using {selected_provider}"):
    if not url:
        st.warning("Please enter a URL.")
    elif not (url.startswith("http://") or url.startswith("https://")):
         st.error("Invalid URL. Please enter a valid HTTP or HTTPS URL.")
    elif not default_model:
         st.error(f"No default model configured for {selected_provider}. Please update `DEFAULT_MODELS` in the code.")
    else:
        with st.spinner(f"Processing with {selected_provider}... Fetching, extracting, analyzing..."):
            # 1. Fetch Content
            response = fetch_content(url)
            extracted_text = None

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
                    extracted_text = extract_html_text(response)
                else:
                    logging.warning(f"Unknown content type '{content_type}'. Attempting HTML extraction for URL: {url}")
                    extracted_text = extract_html_text(response)
                    if not extracted_text:
                         logging.warning(f"HTML extraction failed for unknown type. URL: {url}")
                         st.warning(f"Could not determine content type ('{content_type}'). If this is a PDF, ensure the URL ends with '.pdf'.")


                # 3. Generate Advisory (if text extracted)
                if extracted_text:
                    if len(extracted_text) < 100:
                         st.warning("Extracted text seems very short. The notification might be empty, image-based, or complex. Results may be inaccurate.")
                         logging.warning(f"Extracted text is very short ({len(extracted_text)} chars) for URL: {url}")

                    # Call the main generation function with selected provider and model
                    advisory = generate_compliance_advisory(extracted_text, selected_provider, default_model)

                    # 4. Display Output
                    if advisory:
                        st.success(f"✅ Compliance Advisory Generated Successfully using {selected_provider}!")
                        st.json(advisory) # Display as interactive JSON

                        # Formatted display (remains the same as before)
                        st.subheader("Formatted Advisory Details:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Department:** {advisory.get('Departments', 'N/A')}")
                            st.markdown(f"**Sub-Department:** {advisory.get('Sub-Departments', 'N/A')}")
                            st.markdown(f"**Task Name:** {advisory.get('Task Name', 'N/A')}")
                            st.markdown(f"**Task Description:**")
                            st.markdown(f"> {advisory.get('Task Description', 'N/A')}")
                            st.markdown(f"**Regulatory Risk/Theme:** {advisory.get('Regulatory Risk/Theme', 'N/A')}")
                            st.markdown(f"**Risk Category ID:** {advisory.get('Risk Category ID', 'N/A')}")
                            st.markdown(f"**Inherent Risk Rating:** {advisory.get('Inherent Risk Rating', 'N/A')}")
                            st.markdown(f"**Nature of Mitigation:** {advisory.get('Nature of Mitigation', 'N/A')}")
                        with col2:
                            st.markdown(f"**Compliance Frequency:** {advisory.get('Compliance Frequency', 'N/A')}")
                            st.markdown(f"**Task Frequency Start Date:** {advisory.get('Task Frequency Start Date', 'N/A')}")
                            st.markdown(f"**Task Frequency End Date:** {advisory.get('Task Frequency End Date', 'N/A')}")
                            st.markdown(f"**Due Date:** {advisory.get('Due Date', 'N/A')}")
                            freq = advisory.get('Compliance Frequency')
                            if freq == 'Daily':
                                st.markdown(f"**Daily Occurrence:** Every {advisory.get('D_daysofoccurrence', 'N/A')} day(s)")
                            elif freq == 'Weekly':
                                st.markdown(f"**Weekly Occurrence:** Every {advisory.get('W_weeksofoccurrence', 'N/A')} week(s) on {advisory.get('W_dayofweek', 'N/A')}")
                            elif freq == 'Monthly':
                                st.markdown(f"**Monthly Occurrence:** {advisory.get('M_weeksofoccurrenceformonthly', 'N/A')} {advisory.get('M_dayofweek', 'day')} every {advisory.get('M_monthsofoccurence', 'N/A')} month(s)")
                            elif freq == 'Yearly':
                                st.markdown(f"**Yearly Occurrence:** On {advisory.get('Y_monthofyear', 'N/A')} {advisory.get('Y_dayofmonth', 'N/A')}")

                    else:
                        # Error messages are now handled within generate_compliance_advisory or its sub-functions
                        # A general failure message might still be useful if it returns None without specific errors
                        if 'advisory' not in locals() or advisory is None:
                            st.error(f"💥 Failed to generate compliance advisory using {selected_provider} after extracting text. Check logs or previous messages for details.")
                else:
                    st.error("💥 Failed to extract text content from the URL. Cannot proceed.")
            else:
                # Error handled in fetch_content
                pass

# --- Footer ---
st.markdown("---")
st.markdown("Powered by Streamlit | AI Providers: Gemini, Groq, Mistral, OpenRouter")
