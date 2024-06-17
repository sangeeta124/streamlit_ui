import streamlit as st
from PIL import Image, ImageFont
from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
import os
import numpy as np
import json
import time

# Function to apply PaddleOCR on the uploaded image or PDF page
def apply_paddleocr(image_path):
    ocr = PaddleOCR(use_gpu=False)  # Initialize PaddleOCR with GPU disabled
    result = ocr.ocr(image_path, cls=True)  # Get OCR result with text classification

    if result:  # Check if OCR returned results
        # Load the image using PIL
        image = Image.open(image_path).convert('RGB')

        # Extract boxes, text, and scores
        boxes = [line[0] for line in result[0]]
        txts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        # Specify the path to a .ttf font file
        font_path = "C:\\Windows\\Fonts\\Arial.ttf"  # Update with your valid .ttf font file path

        # Test if the font file can be loaded
        try:
            font = ImageFont.truetype(font_path, 20)
        except IOError:
            st.error(f"Cannot open font file: {font_path}. Check the path.")
            return None, None

        # Draw the results on the image
        image_with_boxes = draw_ocr(np.array(image), boxes, txts, scores, font_path=font_path)  # Convert PIL image to numpy array

        # Convert the result to PIL Image
        image_with_boxes = Image.fromarray(image_with_boxes)

        # Prepare OCR results data for JSON
        ocr_results = [{"text": text, "box": box, "score": score} for text, box, score in zip(txts, boxes, scores)]

        return image_with_boxes, ocr_results
    else:
        st.warning("No text detected by OCR")
        return None, None

# Function to process PDF and convert each page to image
def process_pdf(pdf_path, output_dir):
    images = convert_from_path(pdf_path, dpi=300)  # Convert PDF pages to images at 300 DPI
    pdf_ocr_results = []

    for page_num, page_image in enumerate(images):
        page_image_path = os.path.join(output_dir, f'page_{page_num + 1}.png')
        page_image.save(page_image_path, 'PNG')

        # Apply OCR on the extracted page image
        st.subheader(f'Page {page_num + 1}')
        processed_image, ocr_results = apply_paddleocr(page_image_path)
        if processed_image and ocr_results:
            st.image(processed_image, caption=f"OCR Result Image - Page {page_num + 1}", width=1000)  # Display the processed image
            pdf_ocr_results.append({
                "page": page_num + 1,
                "results": ocr_results
            })

    # Save the combined OCR results for all pages to a JSON file
    json_filename = f'ocr_results_{int(time.time())}.json'
    json_filepath = os.path.join(output_dir, json_filename)
    try:
        with open(json_filepath, 'w', encoding='utf-8') as json_file:
            json.dump(pdf_ocr_results, json_file, ensure_ascii=False, indent=4)
        # Notify the user
        st.success(f"OCR results for PDF saved to {json_filepath}")
        st.write(f"OCR JSON data available at: {json_filepath}")
    except Exception as e:
        st.error(f"Failed to save OCR results for PDF to {json_filepath}: {e}")

# Streamlit code to upload image or PDF and apply OCR
def main():
    st.title("PaddleOCR Demo")

    # File uploader for image or PDF
    uploaded_file = st.file_uploader("Upload Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        temp_file_path = f"./temp_uploaded_file.{file_extension}"
        
        # Save the uploaded file to a temporary location to process it
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Use the directory where the temporary file is stored
        output_dir = os.path.dirname(temp_file_path)

        if file_extension in ["jpg", "jpeg", "png"]:
            # Display the uploaded image
            image = Image.open(temp_file_path)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Button to apply OCR
            if st.button("Apply OCR"):
                # Apply OCR on the uploaded image
                processed_image, ocr_results = apply_paddleocr(temp_file_path)
                if processed_image and ocr_results:
                    # Display the processed image with OCR results
                    st.image(processed_image, caption="OCR Result Image", width=1000)
                    
                    # Save the OCR results to a JSON file
                    json_filename = f'ocr_results_{int(time.time())}.json'
                    json_filepath = os.path.join(output_dir, json_filename)
                    try:
                        with open(json_filepath, 'w', encoding='utf-8') as json_file:
                            json.dump(ocr_results, json_file, ensure_ascii=False, indent=4)
                        # Notify the user
                        st.success(f"OCR results saved to {json_filepath}")
                        st.write(f"OCR JSON data available at: {json_filepath}")
                    except Exception as e:
                        st.error(f"Failed to save OCR results to {json_filepath}: {e}")

        elif file_extension == "pdf":
            # Display a message for PDF
            st.write(f"Uploaded PDF: {uploaded_file.name}")

            # Convert PDF pages to images at 300 DPI for preview
            images = convert_from_path(temp_file_path, dpi=300)
            for page_num, page_image in enumerate(images):
                st.subheader(f'Page {page_num + 1}')
                st.image(page_image, caption=f"Page {page_num + 1}", use_column_width=True)

            # Button to process PDF
            if st.button("Process PDF"):
                # Process PDF and apply OCR on each page
                process_pdf(temp_file_path, output_dir)

if __name__ == "__main__":
    main()


