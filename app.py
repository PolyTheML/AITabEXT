# -*- coding: utf-8 -*-
"""
Streamlit Web App for AI-Powered Table Extraction

This script creates a user-friendly web interface where users can upload an
image of a table, and the application will use an AI model (Gemini) to extract
the data, provide a confidence score, and allow downloads.

To run this app:
1.  Make sure you have all necessary libraries installed:
    pip install streamlit opencv-python numpy pandas requests Pillow python-dotenv openpyxl pandas-gbq google-cloud-bigquery
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
from io import BytesIO
from dotenv import load_dotenv

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="AI Table Extractor & ETL", layout="wide")

# --- Core Logic ---
def prepare_image_from_upload(uploaded_file):
    """Converts and resizes an uploaded image file to a base64 encoded string."""
    try:
        image_bytes = uploaded_file.getvalue()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        max_dim = 2048
        h, w, _ = img_cv.shape
        if h > max_dim or w > max_dim:
            if h > w:
                new_h, new_w = max_dim, int(w * (max_dim / h))
            else:
                new_h, new_w = int(h * (max_dim / w)), max_dim
            img_cv = cv2.resize(img_cv, (new_w, new_h))

        _, buffer = cv2.imencode('.png', img_cv)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    except Exception as e:
        st.error(f"Error preparing image: {e}")
        return None

def extract_table_with_ai(base64_image_data, api_key, model_name):
    """Sends the image to the Gemini API and asks it to extract the table and a confidence score."""
    if not api_key:
        st.error("Error: Gemini API key not found.")
        return None, 0.0, "API key not configured."

    prompt = """
    Analyze the provided image to identify the primary data table. Your task is to extract its content with high precision.
    Instructions:
    1.  **JSON Output:** Return a single JSON object with three keys: "table_data", "confidence_score", and "reasoning".
    2.  **table_data:** The value must be a list of lists, where each inner list represents a table row. The first inner list must be the header.
    3.  **confidence_score:** Provide a numerical score from 0.0 to 1.0, where 1.0 is absolute confidence in the extraction accuracy.
    4.  **reasoning:** Briefly explain your confidence score. Mention any blurry text, complex merged cells, or unusual formatting that might affect accuracy.
    5.  **Accuracy Rules:** Handle merged cells by repeating values, represent empty cells with an empty string(""), and combine multi-line text.
    Begin the extraction now.
    """
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": base64_image_data}}] }],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        json_response_text = result['candidates'][0]['content']['parts'][0]['text']
        parsed_json = json.loads(json_response_text)
        table_data = parsed_json.get("table_data", [])
        confidence = parsed_json.get("confidence_score", 0.0)
        reasoning = parsed_json.get("reasoning", "No reasoning provided.")
        return table_data, confidence, reasoning
    except requests.exceptions.HTTPError as err:
        st.error(f"An HTTP error occurred: {err}")
        st.code(err.response.text, language='json')
        return None, 0.0, "Extraction failed due to an HTTP error."
    except Exception as e:
        st.error(f"An error occurred during extraction: {e}")
        return None, 0.0, "Extraction failed due to a general error."

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def transform_data(df: pd.DataFrame, is_for_viz=False) -> pd.DataFrame:
    """Cleans, standardizes, and validates the extracted DataFrame."""
    if not is_for_viz:
        st.info("Transforming data for ETL...")
    
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            cleaned_col = df[col].str.replace(',', '', regex=False).str.replace('%', '', regex=False).str.strip()
            df[col] = pd.to_numeric(cleaned_col, errors='ignore')
            
        if not pd.api.types.is_numeric_dtype(df[col]):
             try:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
             except (ValueError, TypeError):
                pass
                
    df.replace('', np.nan, inplace=True)
    if not is_for_viz:
        st.success("Data transformation complete!")
    return df

def load_to_bigquery(df: pd.DataFrame, project_id: str, table_id: str):
    try:
        df.to_gbq(destination_table=table_id, project_id=project_id, if_exists='append', progress_bar=True)
        st.success(f"Successfully loaded {len(df)} rows to BigQuery table: {table_id}")
    except Exception as e:
        st.error(f"Failed to load data to BigQuery: {e}")
        st.warning("Ensure your GCP credentials are set up correctly.")

# --- Streamlit App UI ---
def main():
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    st.title("📄 AI-Powered Table Extractor & ETL")
    st.markdown("Upload an image, extract the data, and load it into your data warehouse.")

    if 'extracted_df' not in st.session_state:
        st.session_state.extracted_df = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None
    if 'reasoning' not in st.session_state:
        st.session_state.reasoning = None

    with st.sidebar:
        st.header("⚙️ Options")
        # --- UPDATED MODEL OPTIONS ---
        model_options = [
            "gemini-2.5-pro", 
            "gemini-2.5-flash", 
            "gemini-2.5-flash-lite", 
            "gemini-2.0-flash"
        ]
        selected_model = st.selectbox("Choose your Gemini model:", options=model_options, help="Note: Some models may not be publicly available and could cause errors.")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with col2:
            if st.button("Extract Table from Image", type="primary"):
                with st.spinner(f"The AI is analyzing the image with **{selected_model}**. Please wait..."):
                    base64_image = prepare_image_from_upload(uploaded_file)
                    if base64_image:
                        table_data, confidence, reasoning = extract_table_with_ai(base64_image, gemini_api_key, selected_model)
                        st.session_state.confidence = confidence
                        st.session_state.reasoning = reasoning
                        if table_data and len(table_data) > 1:
                            try:
                                header, data = table_data[0], table_data[1:]
                                new_columns, counts = [], {}
                                for col in header:
                                    counts[col] = counts.get(col, 0) + 1
                                    if counts[col] > 1: new_columns.append(f"{col}_{counts[col]}")
                                    else: new_columns.append(col)
                                st.session_state.extracted_df = pd.DataFrame(data, columns=new_columns)
                                st.success("Table extracted successfully!")
                            except Exception as e:
                                st.error(f"Data Mismatch: {e}. Trying to load without a header.")
                                st.session_state.extracted_df = pd.DataFrame(table_data)
                        elif table_data:
                            st.warning("Only one row of data was extracted.")
                            st.session_state.extracted_df = pd.DataFrame(table_data)
                        else:
                            st.error("The AI could not find a table or the API call failed.")
                            st.session_state.extracted_df = None

    if st.session_state.confidence is not None:
        st.subheader("📊 Extraction Accuracy")
        st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.1%}", delta_color="off")
        with st.expander("See AI's Reasoning"):
            st.info(st.session_state.reasoning)

    if st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
        st.divider()
        st.subheader("Extracted Data Preview")
        st.dataframe(st.session_state.extracted_df)

        st.divider()
        st.subheader("📊 Visualize Your Data")
        if st.checkbox("Show Visualization Options"):
            df_for_viz = transform_data(st.session_state.extracted_df.copy(), is_for_viz=True)
            numeric_columns = df_for_viz.select_dtypes(include=np.number).columns.tolist()
            if not numeric_columns:
                st.warning("No numeric data could be cleaned for plotting.")
            else:
                chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Area Chart"])
                all_columns = df_for_viz.columns.tolist()
                c1, c2 = st.columns(2)
                x_axis = c1.selectbox("Select X-Axis", options=all_columns, key="x_axis")
                y_axis = c2.selectbox("Select Y-Axis (must be numeric)", options=numeric_columns, key="y_axis")
                
                if st.button("Generate Chart", type="primary"):
                    try:
                        if chart_type == "Scatter Plot":
                            st.scatter_chart(df_for_viz, x=x_axis, y=y_axis)
                        else:
                            viz_df = df_for_viz[[x_axis, y_axis]].set_index(x_axis)
                            chart_func = getattr(st, f"{chart_type.lower().replace(' ', '_')}")
                            chart_func(viz_df)
                    except Exception as e:
                        st.error(f"Could not generate chart. Error: {e}")

        st.divider()
        st.subheader("Download Extracted Data")
        file_name_input = st.text_input("Enter your desired filename (without extension)", "extracted_table", key="file_name")
        c1, c2 = st.columns(2)
        csv = st.session_state.extracted_df.to_csv(index=False, encoding='utf-8-sig')
        c1.download_button("Download as CSV", csv, f"{file_name_input}.csv", "text/csv")
        excel_data = to_excel(st.session_state.extracted_df)
        c2.download_button("Download as Excel", excel_data, f"{file_name_input}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.divider()
        st.subheader("🚀 ETL Pipeline: Load to BigQuery")
        project_id = st.text_input("Enter your Google Cloud Project ID", key="project_id")
        table_id = st.text_input("Enter your BigQuery Table ID (e.g., my_dataset.my_table)", key="table_id")
        if st.button("Run Transform & Load to BigQuery"):
            if not project_id or not table_id:
                st.warning("Please provide your Project ID and Table ID to load the data.")
            else:
                with st.spinner("Running ETL process..."):
                    df_to_load = st.session_state.extracted_df.copy()
                    transformed_df = transform_data(df_to_load)
                    load_to_bigquery(transformed_df, project_id, table_id)

if __name__ == '__main__':
    main()