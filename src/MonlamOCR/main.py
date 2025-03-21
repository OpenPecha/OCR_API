from fastapi import FastAPI, HTTPException
import numpy as np
import cv2
import os
import tempfile
import pyewts
import requests
from MonlamOCR.Inference import OCRPipeline
from MonlamOCR.Config import init_monlam_line_model, init_monla_ocr_model
from MonlamOCR.Utils import read_line_model_config
from MonlamOCR.Data import LineData
app = FastAPI()

# Initialize your models and configs
pyewt = pyewts.pyewts()


def initialize_models(OCR_model: str):
    line_model_config = init_monlam_line_model()
    ocr_config = init_monla_ocr_model(OCR_model)
    line_config = read_line_model_config(line_model_config)
    ocr_pipeline = OCRPipeline(
        ocr_config=ocr_config,
        line_config=line_config,
        output_dir="./data/output",
    )
    return ocr_pipeline


def download_image(image_url: str) -> str:
    try:
        # Send a GET request to fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Create a temporary file to save the image
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, "temp_image.jpg")

        # Write the image content to the temporary file
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(response.content)

        # Check if the image can be opened
        image = cv2.imread(temp_file_path)
        if image is None:
            raise ValueError("Could not decode the image from the URL")

        # Return the path to the saved image
        return temp_file_path

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")


def get_line_data_and_page_text(line_inference, image_name):
    line_data = []
    page_text = ""
    for line_num, line_info in enumerate(line_inference, 1):
        line_dict = {
            "line_id": f"{image_name}_{line_num}",
            "line_text": pyewt.toUnicode(line_info['text']) ,
            "line_annotation": line_info['line_annotation']
        }
        line_data.append(line_dict)
        page_text += pyewt.toUnicode(line_info['text']) + "\n"
    return line_data, page_text



@app.post("/process/")
async def process_file(data: dict):
    image_url = data['image_url']
    image_name = image_url.split("/")[-1]
    OCR_model_name = data['OCR_model']
    ocr_pipeline = initialize_models(OCR_model_name)
    try:
        image_path = download_image(image_url)

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image from the saved path.")

        line_inference = ocr_pipeline.run_ocr(image=image)
        line_data, page_text = get_line_data_and_page_text(line_inference, image_name.split(".")[0])

        return {
            "image_name": image_name,
            "OCR_model": {
                        "Name": OCR_model_name,
                        "Version": "v1"
                        },
            "Line_model": {
                        "Name": "PhotiLines",
                        "Version": "v1"
                        },
            "Page_text": page_text,
            "line_data": line_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
