from PIL import ImageTk, Image
import cv2
import os
import subprocess
import json as js
import requests
import pyttsx3

headers = {"Authorization": "Bearer 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"}
######################### MODEL USED ############################################
### OD
OD_URL = "https://api-inference.huggingface.co/models/hustvl/yolos-tiny"

def get_object(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(OD_URL, headers=headers, data=data)
    return response.json()

object = get_object("testimage.jpg")
with open("object.json", "w") as fr:
    js.dump(object, fr,indent=4)
    
### Image Captioning
Image_Captioning_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"

def get_caption(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(Image_Captioning_URL, headers=headers, data=data)
    return response.json()

caption = get_caption("testimage.jpg")
with open("caption.json", "w") as fr:
    js.dump(caption, fr,indent=4)

### Speech recognition
Speech_regcognize_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(Speech_regcognize_URL, headers=headers, data=data)
    return response.json()
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe("")
print(result["text"])

output = query("sample1.flac")

##################################################################################
#  FEATURE 1: GET IMAGE WITH CAM AND INFER, RETURN A VOICE 
##################### PHASE 1: Get command #################################
def capture_image():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if not ret:
        raise RuntimeError("Failed to capture image")
    return frame

def save_image(frame, folder="result"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    image_path = os.path.join(folder, "captured_image.png")
    cv2.imwrite(image_path, frame)
    return image_path



def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def processing():
        try:
            # Capture and save image
            frame = capture_image()
            #cv2.imshow("Captured Image", frame)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            image_path = save_image(frame)
            print(f"Image saved in {image_path}")
            # Run object detection
            par = generate_paragraph(object, output)
            text_to_speech(par)
            print(par)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def display_image(image_path, window_name="Detected Image"):
    image = cv2.imread(image_path)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_paragraph(object_counts, description_list):
    object_descriptions = []
    for obj, count in object_counts.items():
        if count == 1:
            object_descriptions.append(f"1 {obj}")
        else:
            object_descriptions.append(f"{count} {obj}s")
    
    objects_text = ", ".join(object_descriptions[:-1])
    if len(object_descriptions) > 1:
        objects_text += f", and {object_descriptions[-1]}"
    elif len(object_descriptions) == 1:
        objects_text = object_descriptions[0]
    
    paragraph = f"In the picture, there are {objects_text}. "

    if description_list and 'generated_text' in description_list[0]:
        paragraph += description_list[0]['generated_text'].capitalize() + "."
    
    return paragraph