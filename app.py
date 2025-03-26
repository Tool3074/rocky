import streamlit as st
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from pdfminer.high_level import extract_text
import re
import json
import os

# Placeholder for LLM function.  Replace with actual LLM call.
def get_structured_advisory(extracted_text, rbi_url):
    """
    This is a placeholder function.  In a real application, this function
    would call an LLM (like Gemini, GPT-3, etc.) to structure the
    extracted text into the RBI Compliance Advisory format.  This placeholder
    simulates that behavior.

    Args:
        extracted_text (str): The text extracted from the RBI document.
        rbi_url (str): The URL of the RBI document (for context).

    Returns:
        dict: A dictionary representing the structured RBI Compliance Advisory.
              Returns None if it cannot process.
    """
    # Simulate LLM processing and potential failure.
    if not extracted_text:
        return None

    # Basic Heuristics (Improved):
    if "master circular" in extracted_text.lower():
        regulatory_theme = "Governance (Prudential and Licensing)"
    elif "information technology" in extracted_text.lower() or "cybersecurity" in extracted_text.lower():
        regulatory_theme = "Information and Technology"
    elif "customer service" in extracted_text.lower() or "fair practices" in extracted_text.lower():
        regulatory_theme = "Market Conduct and Customers"
    elif "fraud" in extracted_text.lower() or "anti-money laundering" in extracted_text.lower():
        regulatory_theme = "Financial Crime"
    elif "human resources" in extracted_text.lower() or "employee" in extracted_text.lower():
        regulatory_theme = "Employment Practices and Workplace Safety"
    elif "risk management" in extracted_text.lower():
        regulatory_theme = "Risk"
    else:
        regulatory_theme = "Governance (Prudential and Licensing)"  # Default

    # More robust keyword search for task name
    task_name_keywords = ["comply with", "implement", "ensure", "submit", "report", "review", "establish", "maintain", "update", "provide"]
    found_task_name = None
    for keyword in task_name_keywords:
        match = re.search(rf"(?:to\s+)?{keyword}\s+([^\n\r.]*)", extracted_text, re.IGNORECASE)
        if match:
            found_task_name = "To " + match.group(1).strip()
            break
    if not found_task_name:
        found_task_name = "To review the notification" # default

    # Even more robust date extraction
    dates = re.findall(r'(\d{1,2}[-./]\d{1,2}[-./]\d{2,4})', extracted_text) # Finds all date-like strings
    notification_date = None
    if dates:
        notification_date = dates[0]  # Take the first date found.
    else:
        notification_date = "01-04-2024" # Default if no date

    # Create a sample structured advisory (Improved)
    structured_advisory = {
        "Departments": "Administration",
        "Task Name": found_task_name,
        "Task Description": f"Review and implement the guidelines outlined in the RBI notification dated {notification_date}.  Refer to the RBI URL for detailed instructions.",
        "Sub-Departments": "Bankwide",
        "Regulatory Risk/Theme": regulatory_theme,
        "Risk Category ID": "Non Compliance Risk",
        "Inherent Risk Rating": "Medium",
        "Nature of Mitigation": "Policy",
        "Task Frequency Start Date": notification_date,
        "Task Frequency End Date": "31-03-2039",
        "Compliance Frequency": "Onetime",
        "Due Date": "31-03-2099",
        "D_daysofoccurrence": None,
        "W_weeksofoccurrence": None,
        "W_dayofweek": None,
        "M_weeksofoccurrenceformonthly": None,
        "M_dayofweek": None,
        "M_monthsofoccurence": None,
        "Y_monthofyear": None,
        "Y_dayofmonth": None,
    }
    return structured_advisory

def extract_text_from_url(url):
    """
    Extracts text from a given URL, handling both HTML and PDF content.

    Args:
        url (str): The URL to extract text from.

    Returns:
        str: The extracted text, or None on error.
    """
    try:
        response = requests.get(url, timeout=10)  # Increased timeout
        response.raise_for_status()  # Raise HTTPError for bad responses

        if 'text/html' in response.headers['Content-Type']:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Remove script and style tags to get cleaner text
            for script_or_style in soup.find_all(['script', 'style']):
                script_or_style.decompose()
            text = soup.get_text(separator='\n')
            return text
        elif 'application/pdf' in response.headers['Content-Type']:
            pdf_content = BytesIO(response.content)
            text = extract_text(pdf_content)
            return text
        else:
            st.error(f"Unsupported content type: {response.headers['Content-Type']}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing content: {e}")
        return None

def validate_structured_advisory(advisory):
    """
    Validates the structure and content of the generated RBI Compliance Advisory.

    Args:
        advisory (dict): The advisory to validate.

    Returns:
        list: A list of error messages.  Empty list if valid.
    """
    errors = []

    # Check for missing keys
    required_keys = [
        "Departments", "Task Name", "Task Description", "Sub-Departments",
        "Regulatory Risk/Theme", "Risk Category ID", "Inherent Risk Rating",
        "Nature of Mitigation", "Task Frequency Start Date", "Task Frequency End Date",
        "Compliance Frequency", "Due Date", "D_daysofoccurrence",
        "W_weeksofoccurrence", "W_dayofweek", "M_weeksofoccurrenceformonthly",
        "M_dayofweek", "M_monthsofoccurence", "Y_monthofyear", "Y_dayofmonth"
    ]
    for key in required_keys:
        if key not in advisory:
            errors.append(f"Missing required key: {key}")

    # Check for valid values (Example checks.  Extend as needed)
    valid_departments = ["Administration", "Finance & Accounts", "Training", "Procurement & Commercial",
                           "Human Resource", "Information Security (InfoSec)", "Information Technology",
                           "Internal Audit & Risk Management", "Business Operations", "Legal",
                           "Marketing & Communication", "Mergers & Acquisitions", "Pre Sales & Sales",
                           "Quality Assurance"]
    if advisory.get("Departments") not in valid_departments:
        errors.append(f"Invalid Department: {advisory.get('Departments')}")

    valid_sub_departments = ["Operations - Cash Management", "Operations - Trade Services",
                               "Operations - Custody operations", "Operations - Money Market",
                               "Operations - FX", "Operations - ALM", "Operations - Retail Banking",
                               "Operations - Debit/ Credit cards", "Market Risk", "Liquidity Risk",
                               "Credit Risk", "Operational Risk", "Financial Risk", "Bankwide"]
    if advisory.get("Sub-Departments") not in valid_sub_departments:
        errors.append(f"Invalid Sub-Department: {advisory.get('Sub-Departments')}")

    valid_regulatory_themes = ["Governance (Prudential and Licensing)", "Information and Technology",
                                "Market Conduct and Customers", "Financial Crime",
                                "Employment Practices and Workplace Safety", "Process Enhancement",
                                 "Conflict of Interest", "Risk"]
    if advisory.get("Regulatory Risk/Theme") not in valid_regulatory_themes:
        errors.append(f"Invalid Regulatory Risk/Theme: {advisory.get('Regulatory Risk/Theme')}")

    valid_risk_categories = ["Credit & Financial", "Non Implementation Risk", "Credit Regulatory Risk",
                               "Reporting Risk", "Conflict of Interest", "Non Compliance Risk", "IT Security Risk"]
    if advisory.get("Risk Category ID") not in valid_risk_categories:
        errors.append(f"Invalid Risk Category ID: {advisory.get('Risk Category ID')}")

    valid_risk_ratings = ["Very High", "High", "Medium", "Low"]
    if advisory.get("Inherent Risk Rating") not in valid_risk_ratings:
        errors.append(f"Invalid Inherent Risk Rating: {advisory.get('Inherent Risk Rating')}")

    valid_mitigations = ["Policy", "Governance", "Process/controls", "Systems and Technology", "Training"]
    if advisory.get("Nature of Mitigation") not in valid_mitigations:
        errors.append(f"Invalid Nature of Mitigation: {advisory.get('Nature of Mitigation')}")

    valid_frequencies = ["Onetime", "Daily", "Weekly", "Monthly", "Yearly"]
    if advisory.get("Compliance Frequency") not in valid_frequencies:
        errors.append(f"Invalid Compliance Frequency: {advisory.get('Compliance Frequency')}")

    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if advisory.get("W_dayofweek") and advisory.get("W_dayofweek") not in valid_days:
        errors.append(f"Invalid W_dayofweek: {advisory.get('W_dayofweek')}")

    valid_weeks = ["First", "Second", "Third", "Fourth", "Last", "Day"]
    if advisory.get("M_weeksofoccurrenceformonthly") and advisory.get("M_weeksofoccurrenceformonthly") not in valid_weeks:
        errors.append(f"Invalid M_weeksofoccurrenceformonthly: {advisory.get('M_weeksofoccurrenceformonthly')}")

    # Date format validation (basic)
    date_format = r'\d{1,2}[-./]\d{1,2}[-./]\d{2,4}'
    if advisory.get("Task Frequency Start Date") and not re.match(date_format, advisory.get("Task Frequency Start Date")):
        errors.append("Invalid Task Frequency Start Date format. Use DD-MM-YYYY, DD/MM/YYYY, or DD.MM.YYYY")
    if advisory.get("Task Frequency End Date") and not re.match(date_format, advisory.get("Task Frequency End Date")):
        errors.append("Invalid Task Frequency End Date format. Use DD-MM-YYYY, DD/MM/YYYY, or DD.MM.YYYY")
    if advisory.get("Due Date") and not re.match(date_format, advisory.get("Due Date")):
        errors.append("Invalid Due Date format. Use DD-MM-YYYY, DD/MM/YYYY, or DD.MM.YYYY")

    # Check for logical consistency
    if advisory.get("Compliance Frequency") == "Onetime" and advisory.get("Due Date") == "31-03-2099":
        errors.append("For 'Onetime' frequency, Due Date should be a specific date, not 31-03-2099.")

    return errors

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("RBI Compliance Advisory Generator")
    st.markdown("Enter an RBI notification URL to generate a structured compliance advisory.")

    rbi_url = st.text_input("RBI Notification URL:", "")

    if rbi_url:
        with st.spinner("Fetching and processing..."):
            extracted_text = extract_text_from_url(rbi_url)
            if extracted_text:
                structured_advisory = get_structured_advisory(extracted_text, rbi_url)
                if structured_advisory:
                    errors = validate_structured_advisory(structured_advisory)
                    if errors:
                        st.error("Errors in generated advisory:")
                        for error in errors:
                            st.error(error)
                    else:
                        st.subheader("Generated Compliance Advisory:")
                        st.json(structured_advisory)  # Display as JSON for easy viewing
                else:
                    st.error("Failed to generate structured advisory.  The LLM was unable to process the input.")
            else:
                st.error("Failed to extract text from the provided URL.")
    st.markdown("Note: This application is a prototype and may not be fully accurate.  Always verify compliance requirements with the official RBI documentation.")

if __name__ == "__main__":
    main()

