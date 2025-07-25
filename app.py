# -*- coding: utf-8 -*-
"""
Streamlit Web App for AI-Powered Table Extraction

This script creates a user-friendly web interface where users can upload an
image of a table, and the application will use an AI model (Gemini) to extract
the data and provide it as a downloadable CSV or Excel file.

To run this app:
1.  Make sure you have all necessary libraries installed:
    pip install streamlit opencv-python numpy pandas requests Pillow python-dotenv openpyxl
2.  Ensure you have a .env file in the root directory of this project with your
    GEMINI_API_KEY.
3.  Run the app from your terminal:
    streamlit run your_script_name.py
"""

import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import os
import pandas as pd
import base64
import json
import re
from io import BytesIO

# --- Core Logic from Previous Script ---

# FINAL FIX: Manual .env file loader
def load_env_manually(dotenv_path):
    """
    Manually reads and parses a .env file to set environment variables.
    This version reads the file in binary mode and removes null bytes to
    bypass stubborn decoding errors.
    """
    if not os.path.exists(dotenv_path):
        st.error(f"Error: .env file not found at {dotenv_path}. Please create it and add your GEMINI_API_KEY.")
        return False
        
    try:
        with open(dotenv_path, 'rb') as f:
            raw_data = f.read()
        
        sanitized_data = raw_data.replace(b'\x00', b'')
        content = sanitized_data.decode('utf-8-sig', errors='ignore')

        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key] = value
        return True
    except Exception as e:
        st.error(f"An error occurred while manually reading the .env file: {e}")
        return False

# Function to prepare image for API
def prepare_image_from_upload(uploaded_file):
    """Converts an uploaded image file to a base64 encoded string."""
    try:
        # Read the file bytes
        image_bytes = uploaded_file.getvalue()
        # Convert to an OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Encode the image to a PNG format in memory
        _, buffer = cv2.imencode('.png', img_cv)
        # Convert the buffer to a base64 string
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    except Exception as e:
        st.error(f"Error preparing image: {e}")
        return None

# AI Model for Table Extraction
def extract_table_with_ai(base64_image_data, api_key):
    """Sends the image to the Gemini API and asks it to extract the table data."""
    if not api_key:
        st.error("Error: Gemini API key not found. Make sure it's set in your .env file.")
        return None

    prompt = """
    Analyze the provided image. Identify the primary table within it.
    Extract all the data from the table, row by row.
    Return the result as a single JSON object. The object should have a key "table_data"
    which contains a list of lists. Each inner list represents a row from the table,
    and each item in the inner list should be a string representing the text in a cell.
    Handle multi-line text in a cell by combining it into a single string.
    If you cannot find a table, return an empty list.
    """
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64_image_data
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
        }
    }

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        if (result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts')):
            json_response_text = result['candidates'][0]['content']['parts'][0]['text']
            parsed_json = json.loads(json_response_text)
            table_data = parsed_json.get("table_data", [])
            return table_data if table_data else None
        else:
            st.error(f"Error: AI response was not in the expected format. Full response: {result}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during the API request: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from AI response: {e}. Raw response text: {response.text}")
        return None

# Function to convert DataFrame to Excel in memory
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# --- Streamlit App UI ---

def main():
    st.set_page_config(page_title="AI Table Extractor", layout="wide")

    # Load API Key
    # Assumes .env file is in the root directory where you run `streamlit run`
    dotenv_path = os.path.join(os.getcwd(), '.env')
    load_env_manually(dotenv_path)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    st.title("📄 AI-Powered Table Extractor")
    st.markdown("Upload an image containing a table, and the AI will extract the data for you to download.")

    # Initialize session state to hold the DataFrame
    if 'extracted_df' not in st.session_state:
        st.session_state.extracted_df = None

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("Extract Table from Image", type="primary"):
                if not gemini_api_key:
                    st.warning("Please add your Gemini API Key to the .env file before proceeding.")
                else:
                    with st.spinner("The AI is analyzing the image. Please wait..."):
                        # Process the image
                        base64_image = prepare_image_from_upload(uploaded_file)
                        if base64_image:
                            table_data = extract_table_with_ai(base64_image, gemini_api_key)
                            if table_data:
                                try:
                                    # Store DataFrame in session state
                                    st.session_state.extracted_df = pd.DataFrame(table_data)
                                    st.success("Table extracted successfully!")
                                except ValueError:
                                    st.error("Extraction failed. The AI returned rows with different numbers of columns.")
                                    st.session_state.extracted_df = None
                            else:
                                st.error("The AI could not find a table in the image or failed to extract data.")
                                st.session_state.extracted_df = None

    # Display the extracted data and download buttons if a DataFrame exists
    if st.session_state.extracted_df is not None:
        st.subheader("Extracted Data")
        st.dataframe(st.session_state.extracted_df)

        st.subheader("Download Extracted Data")
        col1, col2 = st.columns(2)

        # CSV Download
        csv = st.session_state.extracted_df.to_csv(index=False, encoding='utf-8-sig')
        col1.download_button(
            label="Download as CSV",
            data=csv,
            file_name="extracted_table.csv",
            mime="text/csv",
        )

        # Excel Download
        excel_data = to_excel(st.session_state.extracted_df)
        col2.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name="extracted_table.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == '__main__':
    main()
