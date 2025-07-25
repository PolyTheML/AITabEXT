import streamlit as st
import fitz  # PyMuPDF
import os
import io
import zipfile

# --- Core Conversion Function ---
def convert_pdf_to_images(pdf_file, start_page, end_page, output_folder="pdf_to_images_output"):
    """
    Converts specified pages of a PDF file to PNG images.

    Args:
        pdf_file (BytesIO): The uploaded PDF file.
        start_page (int): The starting page number for conversion.
        end_page (int): The ending page number for conversion.
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
            # Render page to a pixmap (an image)
            pix = page.get_pixmap()
            image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
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

st.title("📄 PDF to Image Converter")
st.write("Upload a PDF file, specify the page range, and convert the pages into high-quality images.")

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

        # --- Page Range Selection ---
        st.header("2. Select Page Range")
        col1, col2 = st.columns(2)
        with col1:
            start_page_input = st.number_input("Start Page", min_value=1, max_value=total_pages, value=1, step=1)
        with col2:
            end_page_input = st.number_input("End Page", min_value=1, max_value=total_pages, value=total_pages, step=1)

        # --- Conversion Button ---
        if st.button("Convert to Images", type="primary"):
            with st.spinner("Converting pages to images... Please wait."):
                # Call the conversion function
                image_files = convert_pdf_to_images(
                    io.BytesIO(pdf_bytes),  # Pass a new BytesIO object
                    start_page_input,
                    end_page_input
                )

            if image_files:
                st.success("✅ Conversion complete!")

                # --- Display Images and Download Link ---
                st.header("3. Converted Images")

                # Create a zip file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for img_path in image_files:
                        zf.write(img_path, os.path.basename(img_path))
                
                # Provide download button for the zip file
                st.download_button(
                    label="📥 Download All Images as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="converted_images.zip",
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
1.  **Upload a PDF:** Click the 'Browse files' button to select a PDF from your computer.
2.  **Select Page Range:** After uploading, input the starting and ending page numbers you want to convert.
3.  **Convert:** Click the 'Convert to Images' button to start the process.
4.  **Download:** Once finished, you can download all images as a single ZIP file or preview them on the page.
""")
