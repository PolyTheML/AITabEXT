# -*- coding: utf-8 -*-
"""
Standalone Streamlit App for High-Quality PDF to Image Conversion

This script creates a user-friendly web interface to convert specific pages
of a PDF document into high-resolution PNG images. The user can adjust the
image quality (DPI) and download the results as a ZIP file.

To run this app:
1.  Make sure you have all necessary libraries installed:
    pip install streamlit "PyMuPDF<1.24.0"
2.  Run the app from your terminal:
    streamlit run PDFconverter.py
"""

import streamlit as st
import fitz  # PyMuPDF
import os
import io
import zipfile

# --- Core Conversion Function ---
def convert_pdf_to_images(pdf_file, start_page, end_page, dpi=300, output_folder="pdf_to_images_output"):
    """
    Converts specified pages of a PDF file to high-resolution PNG images.

    Args:
        pdf_file (BytesIO): The uploaded PDF file.
        start_page (int): The starting page number for conversion.
        end_page (int): The ending page number for conversion.
        dpi (int): The resolution (Dots Per Inch) for the output images.
        output_folder (str): The name of the folder to save images in.

    Returns:
        list: A list of paths to the generated image files.
    """
    image_paths = []
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the PDF from the uploaded file's bytes
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

        # Validate page range
        total_pages = len(pdf_document)
        if start_page > end_page:
            st.error("Error: Start page cannot be greater than the end page.")
            return []
        if start_page < 1 or end_page > total_pages:
            st.error(f"Error: Invalid page range. The PDF has {total_pages} pages.")
            return []

        # Convert selected pages to images
        for page_num in range(start_page - 1, end_page):
            page = pdf_document.load_page(page_num)
            
            # Create a matrix for zooming, which corresponds to the DPI
            # A DPI of 72 is standard, so we scale from there.
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page to a pixmap (an image) using the zoom matrix
            pix = page.get_pixmap(matrix=mat)
            
            image_path = os.path.join(output_folder, f"page_{page_num + 1}_dpi{dpi}.png")
            
            # Save the pixmap as a PNG file
            pix.save(image_path)
            image_paths.append(image_path)

        pdf_document.close()
        return image_paths

    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        return []

# --- Streamlit App UI ---

st.set_page_config(layout="centered", page_title="PDF to Image Converter")

st.title("📄 High-Quality PDF to Image Converter")
st.markdown("Upload a PDF file, specify the page range and image quality, and convert the pages into clear images.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Display PDF info once uploaded
    try:
        pdf_bytes = uploaded_file.getvalue()
        pdf_doc_info = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        total_pages = len(pdf_doc_info)
        pdf_doc_info.close()

        st.info(f"PDF uploaded successfully! It has **{total_pages}** pages.")

        # --- Page Range and Quality Selection ---
        st.header("Select Page Range and Quality")
        col1, col2 = st.columns(2)
        with col1:
            start_page_input = st.number_input("Start Page", min_value=1, max_value=total_pages, value=1, step=1)
        with col2:
            end_page_input = st.number_input("End Page", min_value=1, max_value=total_pages, value=total_pages, step=1)
        
        # DPI slider for image quality
        dpi_input = st.slider(
            "Image Quality (DPI)", 
            min_value=100, 
            max_value=600, 
            value=300, 
            step=50,
            help="Higher DPI means clearer images, ideal for OCR. 300 is recommended."
        )

        # --- Conversion Button ---
        if st.button("Convert to Images", type="primary"):
            with st.spinner("Converting pages to high-quality images... Please wait."):
                # Call the conversion function
                image_files = convert_pdf_to_images(
                    io.BytesIO(pdf_bytes),  # Pass a new BytesIO object
                    start_page_input,
                    end_page_input,
                    dpi_input
                )

            if image_files:
                st.success("✅ Conversion complete!")

                # --- Display Images and Download Link ---
                st.header("Converted Images")

                # Create a zip file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for img_path in image_files:
                        zf.write(img_path, os.path.basename(img_path))
                
                # Provide download button for the zip file
                st.download_button(
                    label="📥 Download All Images as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"converted_images_dpi{dpi_input}.zip",
                    mime="application/zip",
                )

                # Display a preview of the generated images
                st.write("Image Previews:")
                for img_path in image_files:
                    st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)

    except Exception as e:
        st.error(f"Failed to read the PDF file. It might be corrupted or protected. Error: {e}")

# --- Instructions and Footer ---
st.markdown("---")
st.subheader("How to Use:")
st.markdown("""
1.  **Upload a PDF:** Click the 'Browse files' button to select a PDF.
2.  **Select Options:** Input the page range and adjust the DPI slider for desired image quality.
3.  **Convert:** Click the 'Convert to Images' button.
4.  **Download:** Once finished, download all images as a single ZIP file.
""")
