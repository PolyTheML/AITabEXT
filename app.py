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

def prepare_image_from_upload(uploaded_file):
    """Converts and resizes an uploaded image file to a base64 encoded string."""
    try:
        image_bytes = uploaded_file.getvalue()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- Optimization: Resize Image ---
        max_dim = 2048
        h, w, _ = img_cv.shape
        if h > max_dim or w > max_dim:
            if h > w:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            else:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            img_cv = cv2.resize(img_cv, (new_w, new_h))
        # --- End of Optimization ---

        _, buffer = cv2.imencode('.png', img_cv)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    except Exception as e:
        st.error(f"Error preparing image: {e}")
        return None

def extract_table_with_ai(base64_image_data, api_key, model_name):
    """Sends the image to the Gemini API and asks it to extract the table data."""
    if not api_key:
        st.error("Error: Gemini API key not found. Make sure it's set in your .env file or Streamlit secrets.")
        return None

    # --- Optimized Prompt ---
    prompt = """
    Analyze the provided image to identify the primary data table. Your task is to extract its content with high precision.

    Instructions:
    1.  **JSON Output:** Return a single JSON object with one key: "table_data".
    2.  **Structure:** The value of "table_data" must be a list of lists, where each inner list represents a table row.
    3.  **Header First:** The very first inner list must be the table's header row.
    4.  **Handle Merged Cells:** If a cell spans multiple rows or columns, repeat its value in each corresponding cell of the output.
    5.  **Empty Cells:** Represent any visually empty cells as an empty string ("").
    6.  **Multi-line Text:** Combine text from multiple lines within a single cell into one string, using a space as a separator.
    7.  **Accuracy is Key:** Do not invent data. If you cannot find a table, return an empty list for "table_data".

    Begin the extraction now.
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

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        json_response_text = result['candidates'][0]['content']['parts'][0]['text']
        parsed_json = json.loads(json_response_text)
        table_data = parsed_json.get("table_data", [])
        return table_data if table_data else None

    except requests.exceptions.HTTPError as err:
        st.error(f"An HTTP error occurred: {err}")
        st.code(err.response.text, language='json')
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

def to_excel(df):
    """Converts a DataFrame to an in-memory Excel file."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def make_columns_unique(columns):
    """Appends a suffix to duplicate column names to make them unique."""
    seen = {}
    new_columns = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    return new_columns

# --- Streamlit App UI ---

def main():
    st.set_page_config(page_title="AI Table Extractor", layout="wide")

    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    st.title("📄 AI-Powered Table Extractor")
    st.markdown("Upload an image, choose a model, and the AI will extract the data for you.")

    with st.sidebar:
        st.header("⚙️ Options")
        
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

    if 'extracted_df' not in st.session_state:
        st.session_state.extracted_df = None

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            # FIX: Changed use_column_width to use_container_width
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.button("Extract Table from Image", type="primary"):
                if not gemini_api_key:
                    st.warning("Please add your Gemini API Key to a `.env` file in the project directory.")
                else:
                    with st.spinner(f"The AI is analyzing the image with **{selected_model}**. Please wait..."):
                        base64_image = prepare_image_from_upload(uploaded_file)
                        if base64_image:
                            table_data = extract_table_with_ai(base64_image, gemini_api_key, selected_model)
                            if table_data and len(table_data) > 1:
                                try:
                                    # FIX: Ensure column headers are unique before creating the DataFrame
                                    header = table_data[0]
                                    unique_header = make_columns_unique(header)
                                    data = table_data[1:]
                                    st.session_state.extracted_df = pd.DataFrame(data, columns=unique_header)
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

    if st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.subheader("Extracted Data Preview")
        st.dataframe(st.session_state.extracted_df)

        st.subheader("Download Extracted Data")

        file_name_input = st.text_input("Enter your desired filename (without extension)", "extracted_table")

        col1_dl, col2_dl = st.columns(2)

        csv = st.session_state.extracted_df.to_csv(index=False, encoding='utf-8-sig')
        col1_dl.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"{file_name_input}.csv",
            mime="text/csv",
        )

        excel_data = to_excel(st.session_state.extracted_df)
        col2_dl.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name=f"{file_name_input}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == '__main__':
    main()