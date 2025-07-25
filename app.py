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
from dotenv import load_dotenv

# --- Core Logic from Previous Script ---

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
# Updated to accept a model_name parameter
def extract_table_with_ai(base64_image_data, api_key, model_name):
    """Sends the image to the Gemini API and asks it to extract the table data."""
    if not api_key:
        st.error("Error: Gemini API key not found. Make sure it's set in your .env file or Streamlit secrets.")
        return None

    prompt = """
    Analyze the provided image. Identify the primary table within it.
    Extract all the data from the table, row by row.
    Return the result as a single JSON object. The object should have a key "table_data"
    which contains a list of lists. Each inner list represents a row from the table,
    and each item in the inner list should be a string representing the text in a cell.
    Handle multi-line text in a cell by combining it into a single string.
    If you cannot find a table, return an empty list.
    The first inner list should be the table's header row.
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

    # Use the selected model name to build the API URL
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        json_response_text = result['candidates'][0]['content']['parts'][0]['text']
        parsed_json = json.loads(json_response_text)
        table_data = parsed_json.get("table_data", [])
        return table_data if table_data else None

    except requests.exceptions.HTTPError as err:
        st.error(f"An HTTP error occurred: {err}")
        st.code(err.response.text, language='json') # Show the actual error response from the API
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred during the API request: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from AI response: {e}. Raw response text was: {json_response_text}")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"AI response was not in the expected format. Full response: {result}")
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

    # Load environment variables.
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    st.title("📄 AI-Powered Table Extractor")
    st.markdown("Upload an image, choose a model, and the AI will extract the data for you.")

    # --- Sidebar for Options ---
    with st.sidebar:
        st.header("⚙️ Options")
        
        # Add a selectbox for the user to choose the model
        model_options = [
            "gemini-2.5-pro", 
            "gemini-2.5-flash", 
            "gemini-2.5-flash-lite", 
            "gemini-2.0-flash"
        ]
        selected_model = st.selectbox(
            "Choose your Gemini model:",
            options=model_options,
            help="**Flash** is faster and cheaper. **Pro** is more powerful and accurate."
        )

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
                    st.warning("Please add your Gemini API Key to a `.env` file in the project directory.")
                else:
                    with st.spinner(f"The AI is analyzing the image with **{selected_model}**. Please wait..."):
                        base64_image = prepare_image_from_upload(uploaded_file)
                        if base64_image:
                            # Pass the selected model to the extraction function
                            table_data = extract_table_with_ai(base64_image, gemini_api_key, selected_model)
                            if table_data and len(table_data) > 1:
                                try:
                                    header = table_data[0]
                                    data = table_data[1:]
                                    st.session_state.extracted_df = pd.DataFrame(data, columns=header)
                                    st.success("Table extracted successfully!")
                                except (ValueError, IndexError):
                                    st.error("Data Mismatch: Trying to load without a header.")
                                    try:
                                        st.session_state.extracted_df = pd.DataFrame(table_data)
                                    except Exception as ex:
                                        st.error(f"Fallback failed. Could not create DataFrame. Error: {ex}")
                                        st.session_state.extracted_df = None
                            elif table_data:
                                 st.warning("Only one row of data was extracted. Displaying without a header.")
                                 st.session_state.extracted_df = pd.DataFrame(table_data)
                            else:
                                st.error("The AI could not find a table in the image or the API call failed.")
                                st.session_state.extracted_df = None

    # Display the extracted data and download options if a DataFrame exists
    if st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.subheader("Extracted Data Preview")
        st.dataframe(st.session_state.extracted_df)

        st.subheader("Download Extracted Data")

        # Add a text input for the user to name the file
        file_name_input = st.text_input("Enter your desired filename (without extension)", "extracted_table")

        col1_dl, col2_dl = st.columns(2)

        # CSV Download
        csv = st.session_state.extracted_df.to_csv(index=False, encoding='utf-8-sig')
        col1_dl.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"{file_name_input}.csv", # Use the user's input for the filename
            mime="text/csv",
        )

        # Excel Download
        excel_data = to_excel(st.session_state.extracted_df)
        col2_dl.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"{file_name_input}.xlsx", # Use the user's input for the filename
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == '__main__':
    main()