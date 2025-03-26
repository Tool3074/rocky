import streamlit as st
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from pdfminer.high_level import extract_text
import re
import json
from typing import Optional, Dict, List
import google.generativeai as genai  # For Gemini
import openai  # For OpenAI (Openrouter)
from dotenv import load_dotenv  # For API key management
import logging  # For enhanced logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s"
)


load_dotenv()  # Load environment variables from .env file


# Initialize LLM clients (Add your API keys as secrets in Streamlit)
# For Gemini
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-pro")  # Or "gemini-pro-vision"
    logging.info("Gemini client initialized.")
except KeyError:
    st.warning(
        "Gemini API key not found in Streamlit secrets. Gemini functionality will be limited."
    )
    gemini_model = None
    logging.warning("Gemini API key not found.")


# For Openrouter (OpenAI)
try:
    openai.api_key = st.secrets["OPENROUTER_API_KEY"]
    openai.api_base = "https://openrouter.ai/api/v1"
    logging.info("Openrouter client initialized.")
except KeyError:
    st.warning(
        "Openrouter API key not found in Streamlit secrets. Openrouter functionality will be limited."
    )
    openai = None
    logging.warning("Openrouter API key not found.")


def get_structured_advisory(
    extracted_text: str, rbi_url: str, llm_model: str
) -> Optional[Dict]:
    """
    This function calls an LLM to structure the extracted text into the
    RBI Compliance Advisory format.

    Args:
        extracted_text (str): The text extracted from the RBI document.
        rbi_url (str): The URL of the RBI document (for context).
        llm_model (str): The LLM model to use ("gemini", "openrouter").

    Returns:
        Optional[Dict]: A dictionary representing the structured RBI Compliance Advisory,
                        or None on error.
    """
    if not extracted_text:
        logging.error("No text extracted from the RBI document.")
        return None

    prompt = f"""
    You are an expert compliance officer. Analyze the following text from an RBI notification and structure it into a compliance advisory.
    
    Text:
    {extracted_text}
    
    URL: {rbi_url}
    
    Provide the output in the following JSON format.  Strictly adhere to this format.  If a field cannot be determined from the text, use null, but do not omit any fields.  Do not add any extra text or explanation before or after the JSON.
    
    {{
        "Departments": (Select one from: Administration, Finance & Accounts, Training, Procurement & Commercial, Human Resource, Information Security (InfoSec), Information Technology, Internal Audit & Risk Management, Business Operations, Legal, Marketing & Communication, Mergers & Acquisitions, Pre Sales & Sales, Quality Assurance),
        "Task Name": (Briefly state the task action, starting with "To...".  If no explicit task, use "To review the notification"),
        "Task Description": (Clearly and concisely describe the required action, risk mitigation, or bank procedure in simple language.),
        "Sub-Departments": (Select one from: Operations - Cash Management, Operations - Trade Services, Operations - Custody operations, Operations - Money Market, Operations - FX, Operations - ALM, Operations - Retail Banking, Operations - Debit/ Credit cards, Market Risk, Liquidity Risk, Credit Risk, Operational Risk, Financial Risk, Bankwide),
        "Regulatory Risk/Theme": (Select one from: Governance (Prudential and Licensing), Information and Technology, Market Conduct and Customers, Financial Crime, Employment Practices and Workplace Safety, Process Enhancement, Conflict of Interest, Risk),
        "Risk Category ID": (Select one from: Credit & Financial, Non Implementation Risk, Credit Regulatory Risk, Reporting Risk, Conflict of Interest, Non Compliance Risk, IT Security Risk),
        "Inherent Risk Rating": (Select one from: Very High, High, Medium, Low),
        "Nature of Mitigation": (Select one from: Policy, Governance, Process/controls, Systems and Technology, Training),
        "Task Frequency Start Date": (Use '01-04-2024' for recurring tasks, or the 'Notification date' for non-recurring tasks.  Use format DD-MM-YYYY),
        "Task Frequency End Date": (Use '31-03-2039'. Use format DD-MM-YYYY),
        "Compliance Frequency": (Select one from: Onetime, Daily, Weekly, Monthly, Yearly),
        "Due Date": (Use '31-03-2099' for one-time tasks. Use format DD-MM-YYYY),
        "D_daysofoccurrence": (If Compliance Frequency is 'Daily', specify the occurrence interval, otherwise null),
        "W_weeksofoccurrence": (If Compliance Frequency is 'Weekly', specify the occurrence interval, otherwise null),
        "W_dayofweek": (Select one from: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, otherwise null),
        "M_weeksofoccurrenceformonthly": (Select one from: First, Second, Third, Fourth, Last, Day, otherwise null),
        "M_dayofweek": (Specify the day of the month for 'Monthly' frequency, otherwise null),
        "M_monthsofoccurence": (Specify the monthly occurrence interval, otherwise null),
        "Y_monthofyear": (Specify the month for 'Yearly' frequency, otherwise null),
        "Y_dayofmonth": (Specify the day of the month for 'Yearly' frequency, otherwise null)
    }}
    """

    try:
        if llm_model == "gemini" and gemini_model:
            try:
                response = gemini_model.generate_text(prompt=prompt)
                if response and response.text:
                    return json.loads(response.text)
                else:
                    logging.warning("Gemini returned an empty response.")
                    return None
            except Exception as e:
                logging.error(f"Gemini API error: {e}")
                st.error(f"Gemini API error: {e}")
                return None

        elif llm_model == "openrouter" and openai:
            try:
                response = openai.Completion.create(
                    model="openai/gpt-3.5-turbo",  # Or another Openrouter model
                    prompt=prompt,
                    max_tokens=1000,  # Adjust as needed
                    # Add headers for Openrouter
                    headers={
                        "HTTP-Referer": "https://your-website.com",  # Replace with your website
                        "X-Title": "RBI Compliance Tool",  # Replace with your app's title
                    },
                )
                if response and response.choices and response.choices[0].text:
                    return json.loads(response.choices[0].text.strip())
                else:
                    logging.warning("Openrouter returned an empty response.")
                    return None
            except Exception as e:
                logging.error(f"Openrouter API error: {e}")
                st.error(f"Openrouter API error: {e}")
                return None

        else:
            error_message = f"LLM model {llm_model} is either unsupported or its API key is missing."
            logging.error(error_message)
            raise ValueError(error_message)
    except ValueError as ve:
        st.error(ve)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.exception(f"An unexpected error occurred: {e}")  # Log the full traceback
        return None


def extract_text_from_url(url: str) -> Optional[str]:
    """
    Extracts text from a given URL, handling both HTML and PDF content.

    Args:
        url (str): The URL to extract text from.

    Returns:
        Optional[str]: The extracted text, or None on error.
    """
    try:
        response = requests.get(url, timeout=15)  # Increased timeout
        response.raise_for_status()  # Raise HTTPError for bad responses
        content_type = response.headers["Content-Type"]

        if "text/html" in content_type:
            soup = BeautifulSoup(response.content, "html.parser")
            # Remove script and style tags to get cleaner text
            for script_or_style in soup.find_all(["script", "style"]):
                script_or_style.decompose()
            text = soup.get_text(separator="\n")
            logging.info(f"Successfully extracted text from HTML URL: {url}")
            return text
        elif "application/pdf" in content_type:
            pdf_content = BytesIO(response.content)
            text = extract_text(pdf_content)
            logging.info(f"Successfully extracted text from PDF URL: {url}")
            return text
        else:
            error_message = f"Unsupported content type: {content_type} for URL: {url}"
            st.error(error_message)
            logging.error(error_message)
            return None
    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching URL: {e} for URL: {url}"
        st.error(error_message)
        logging.error(error_message)
        return None
    except Exception as e:
        error_message = f"Error processing content: {e} for URL: {url}"
        st.error(error_message)
        logging.exception(
            error_message
        )  # Log the full traceback, including exception type, stack trace
        return None


def validate_structured_advisory(advisory: Dict) -> List[str]:
    """
    Validates the structure and content of the generated RBI Compliance Advisory.

    Args:
        advisory (dict): The advisory to validate.

    Returns:
        List[str]: A list of error messages.  Empty list if valid.
    """
    errors = []

    if not isinstance(advisory, dict):
        errors.append("Advisory is not a dictionary.")
        return errors  # Stop further checks if it's not a dict

    # Check for missing keys
    required_keys = [
        "Departments",
        "Task Name",
        "Task Description",
        "Sub-Departments",
        "Regulatory Risk/Theme",
        "Risk Category ID",
        "Inherent Risk Rating",
        "Nature of Mitigation",
        "Task Frequency Start Date",
        "Task Frequency End Date",
        "Compliance Frequency",
        "Due Date",
        "D_daysofoccurrence",
        "W_weeksofoccurrence",
        "W_dayofweek",
        "M_weeksofoccurrenceformonthly",
        "M_dayofweek",
        "M_monthsofoccurence",
        "Y_monthofyear",
        "Y_dayofmonth",
    ]
    for key in required_keys:
        if key not in advisory:
            errors.append(f"Missing required key: {key}")

    if errors:
        return errors  # Return early if there are missing keys

    # Check for valid values (Example checks.  Extend as needed)
    valid_departments = [
        "Administration",
        "Finance & Accounts",
        "Training",
        "Procurement & Commercial",
        "Human Resource",
        "Information Security (InfoSec)",
        "Information Technology",
        "Internal Audit & Risk Management",
        "Business Operations",
        "Legal",
        "Marketing & Communication",
        "Mergers & Acquisitions",
        "Pre Sales & Sales",
        "Quality Assurance",
    ]
    if advisory.get("Departments") not in valid_departments:
        errors.append(f"Invalid Department: {advisory.get('Departments')}")

    valid_sub_departments = [
        "Operations - Cash Management",
        "Operations - Trade Services",
        "Operations - Custody operations",
        "Operations - Money Market",
        "Operations - FX",
        "Operations - ALM",
        "Operations - Retail Banking",
        "Operations - Debit/ Credit cards",
        "Market Risk",
        "Liquidity Risk",
        "Credit Risk",
        "Operational Risk",
        "Financial Risk",
        "Bankwide",
    ]
    if advisory.get("Sub-Departments") not in valid_sub_departments:
        errors.append(f"Invalid Sub-Department: {advisory.get('Sub-Departments')}")

    valid_regulatory_themes = [
        "Governance (Prudential and Licensing)",
        "Information and Technology",
        "Market Conduct and Customers",
        "Financial Crime",
        "Employment Practices and Workplace Safety",
        "Process Enhancement",
        "Conflict of Interest",
        "Risk",
    ]
    if advisory.get("Regulatory Risk/Theme") not in valid_regulatory_themes:
        errors.append(
            f"Invalid Regulatory Risk/Theme: {advisory.get('Regulatory Risk/Theme')}"
        )

    valid_risk_categories = [
        "Credit & Financial",
        "Non Implementation Risk",
        "Credit Regulatory Risk",
        "Reporting Risk",
        "Conflict of Interest",
        "Non Compliance Risk",
        "IT Security Risk",
    ]
    if advisory.get("Risk Category ID") not in valid_risk_categories:
        errors.append(f"Invalid Risk Category ID: {advisory.get('Risk Category ID')}")

    valid_risk_ratings = ["Very High", "High", "Medium", "Low"]
    if advisory.get("Inherent Risk Rating") not in valid_risk_ratings:
        errors.append(
            f"Invalid Inherent Risk Rating: {advisory.get('Inherent Risk Rating')}"
        )

    valid_mitigations = [
        "Policy",
        "Governance",
        "Process/controls",
        "Systems and Technology",
        "Training",
    ]
    if advisory.get("Nature of Mitigation") not in valid_mitigations:
        errors.append(f"Invalid Nature of Mitigation: {advisory.get('Nature of Mitigation')}")

    valid_frequencies = ["Onetime", "Daily", "Weekly", "Monthly", "Yearly"]
    if advisory.get("Compliance Frequency") not in valid_frequencies:
        errors.append(f"Invalid Compliance Frequency: {advisory.get('Compliance Frequency')}")

    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if advisory.get("W_dayofweek") and advisory.get("W_dayofweek") not in valid_days:
        errors.append(f"Invalid W_dayofweek: {advisory.get('W_dayofweek')}")

    valid_weeks = ["First", "Second", "Third", "Fourth", "Last", "Day"]
    if (
        advisory.get("M_weeksofoccurrenceformonthly")
        and advisory.get("M_weeksofoccurrenceformonthly") not in valid_weeks
    ):
        errors.append(
            f"Invalid M_weeksofoccurrenceformonthly: {advisory.get('M_weeksofoccurrenceformonthly')}"
        )

    # Date format validation (basic)
    date_format = r"\d{1,2}[-./]\d{1,2}[-./]\d{2,4}"
    if (
        advisory.get("Task Frequency Start Date")
        and not re.match(date_format, advisory.get("Task Frequency Start Date"))
    ):
        errors.append(
            "Invalid Task Frequency Start Date format. Use DD-MM-YYYY, DD/MM/YYYY, or DD.MM.YYYY"
        )
    if (
        advisory.get("Task Frequency End Date")
        and not re.match(date_format, advisory.get("Task Frequency End Date"))
    ):
        errors.append(
            "Invalid Task Frequency End Date format. Use DD-MM-YYYY, DD/MM/YYYY, or DD.MM.YYYY"
        )
    if (
        advisory.get("Due Date")
        and not re.match(date_format, advisory.get("Due Date"))
    ):
        errors.append(
            "Invalid Due Date format. Use DD-MM-YYYY, DD/MM/YYYY, or DD.MM.YYYY"
        )

    # Check for logical consistency
    if (
        advisory.get("Compliance Frequency") == "Onetime"
        and advisory.get("Due Date") == "31-03-2099"
    ):
        errors.append(
            "For 'Onetime' frequency, Due Date should be a specific date, not 31-03-2099."
        )

    return errors


def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("RBI Compliance Advisory Generator")
    st.markdown("Enter an RBI notification URL to generate a structured compliance advisory.")

    rbi_url = st.text_input("RBI Notification URL:", "")
    llm_model = st.selectbox(
        "Choose LLM Model", ["gemini", "openrouter"]
    )  # removed groq and mistral

    if rbi_url:
        with st.spinner("Fetching and processing..."):
            extracted_text = extract_text_from_url(rbi_url)
            if extracted_text:
                structured_advisory = get_structured_advisory(
                    extracted_text, rbi_url, llm_model
                )  # Pass the model
                if structured_advisory:
                    errors = validate_structured_advisory(structured_advisory)
                    if errors:
                        st.error("Errors in generated advisory:")
                        for error in errors:
                            st.error(error)
                    else:
                        st.subheader("Generated Compliance Advisory:")
                        st.json(
                            structured_advisory
                        )  # Display as JSON for easy viewing
                        logging.info("Successfully generated and validated advisory.")
                else:
                    st.error(
                        "Failed to generate structured advisory.  The LLM was unable to process the input."
                    )
                    logging.error("LLM failed to generate structured advisory.")
            else:
                st.error("Failed to extract text from the provided URL.")
                logging.error("Failed to extract text from URL.")

    st.markdown(
        "Note: This application is a prototype and may not be fully accurate.  Always verify compliance requirements with the official RBI documentation."
    )


if __name__ == "__main__":
    main()
