import requests
from pathlib import Path

def send_request_to_api(url, data):
    try:
        response = requests.post(url, json=data)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": "An error occurred while making the request", "details": str(e)}

def write_json(data, image_group):
    with open(f"./data/{image_group}/{data["image_name"]}.json", "w", encoding='utf-8') as f:
        f.write(str(data))
    



def main():
    for image_dir in Path("/Users/tashitsering/Desktop/work/OCR-Deployment/W30199").iterdir():
        vol_text = ""
        for image_path in (image_dir / "images").iterdir():
            if image_path.suffix == ".json":
                continue
            image_group = image_path.parent.parents[0].name
            if Path(f"./data/{image_group}/{image_path.stem}.txt").exists():
                continue
            data = {
            "image_path": image_path.__str__(),
            "OCR_model": "Woodblock"
            }
            response = send_request_to_api("http://127.0.0.1:8000/process/", data)
            
            if not Path(f"./data/{image_group}").exists():
                Path(f"./data/{image_group}").mkdir()
            Path(f"./data/{image_group}/{response["image_name"]}.txt").write_text(response["Page_text"], encoding='utf-8')
        #     vol_text += response["Page_text"]
        # Path(f"./data/{image_group}.txt").write_text(vol_text, encoding='utf-8')
    

if __name__ == "__main__":
    main()


# Some example images to test the API 
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0001.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0002.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0003.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/I3CN78390293.png