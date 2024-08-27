import requests

def download_weights_from_url(url, output_filename=None):
    
    try:
        response = requests.get(url)
        with open(output_filename, 'wb') as file:
            file.write(response.content)

        print(f"File downloaded and saved as {output_filename}")
    except Exception as error:
        raise RuntimeError(f"[ERROR] There is no such url: {url} - error")
    
if __name__ == '__main__':
    url = "https://huggingface.co/spaces/hamhanry/YOLOv10-OBB/resolve/main/pretrained/yolov10s-obb.pt"
    output_filename = "/HDD/etc/yolov10s-obb.pt"
    download_weights_from_url(url, output_filename)