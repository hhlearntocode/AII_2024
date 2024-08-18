from PIL import ImageTk, Image
import cv2
import os
import subprocess
import json as js
import requests
# import pyttsx3
import pyaudio
import speech_recognition as sr
import wave
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound
import tempfile
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
headers = {"Authorization": "Bearer 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"}
######################### MODEL USED ############################################
### OD
OD_URL = "https://api-inference.huggingface.co/models/hustvl/yolos-tiny"

def get_object(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(OD_URL, headers=headers, data=data)
    return response.json()

def get_obj_json(filename):
    os.makedirs("json", exist_ok=True)
    object = get_object(filename)
    with open("json/object.json", "w") as fr:
        js.dump(object, fr,indent=4)
    
### Image Captioning
Image_Captioning_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"

def get_caption(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(Image_Captioning_URL, headers=headers, data=data)
    return response.json()

def get_cap_json(filename):
    os.makedirs("json", exist_ok=True)
    caption = get_caption(filename)
    with open("json/caption.json", "w") as fr:
        js.dump(caption, fr,indent=4)

### Speech recognition
Speech_regcognize_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

def det_speech(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(Speech_regcognize_URL, headers=headers, data=data)
    return response.json()


def get_det_json(filename):
    os.makedirs("json", exist_ok=True)
    audio = det_speech(filename)
    with open("json/audio.json", "w") as fr:
        js.dump(audio, fr,indent=4)

##################################################################################
#       FEATURE 1: GET IMAGE WITH CAM AND INFER, RETURN A VOICE DESCRIBING       #
##################################################################################
def capture_image():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    if not ret:
        raise RuntimeError("Failed to capture image")
    return frame

def save_image(frame, folder="image"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    image_path = os.path.join(folder, "captured_image.png")
    cv2.imwrite(image_path, frame)
    return image_path

def text_to_speech(text, lang='vi'):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_filename = fp.name
    
    tts = gTTS(text=text, lang=lang)
    tts.save(temp_filename)
    
    playsound(temp_filename)
    
    os.remove(temp_filename)

def display_image(image_path, window_name="Detected Image"):
    image = cv2.imread(image_path)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_image_description(json_folder="json"):
    def read_json_file(file_path):
        with open(file_path, 'r') as file:
            return js.load(file)

    caption_file = os.path.join(json_folder, "caption.json")
    object_file = os.path.join(json_folder, "object.json")

    translator = GoogleTranslator(source='auto', target='vi')

    try:
        caption_data = read_json_file(caption_file)
        object_data = read_json_file(object_file)

        object_counts = {}
        for obj in object_data:
            if isinstance(obj, dict) and 'label' in obj:
                label = obj['label']
                translated_label = translator.translate(label)
                object_counts[translated_label] = object_counts.get(translated_label, 0) + 1
            else:
                raise ValueError("Dữ liệu đối tượng không có 'label' hoặc không phải là dictionary")

        object_descriptions = [f"{count} {obj}" for obj, count in object_counts.items()]

        if len(object_descriptions) > 1:
            objects_text = ", ".join(object_descriptions[:-1]) + f" và {object_descriptions[-1]}"
        else:
            objects_text = object_descriptions[0] if object_descriptions else ""

        paragraph = f"Trong hình có {objects_text}. "
    except (IOError, ValueError, KeyError) as e:
        paragraph = "Đã có lỗi xảy ra trong quá trình nhận diện vật thể. "
    
    if caption_data and isinstance(caption_data, list) and 'generated_text' in caption_data[0]:
        english_caption = caption_data[0]['generated_text']
        vietnamese_caption = translator.translate(english_caption)
        paragraph += vietnamese_caption.capitalize() + "."
    else:
        paragraph += "Không thể tạo mô tả cho hình ảnh."

    return paragraph



def process_feat1():
        try:
            frame = capture_image()
            image_path = save_image(frame)
            print(f"Image saved in {image_path}")
            get_obj_json(image_path)
            get_cap_json(image_path)
            par = generate_image_description()
            text_to_speech(par)
            print(par)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

#################################################################################################
#       FEATURE 2: INPUT COMMAND, GET IMAGE WITH CAM AND INFER, RETURN A VOICE DESCRIBING       #
#################################################################################################

def record_audio(filename, duration=5, sample_rate=44100, channels=2, chunk=1024):
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
    
    print("Recording...")
    
    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save as WAV first
    with wave.open(filename + ".wav", 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))


def listen_and_recognize(recognizer, microphone):
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
        try:
            return recognizer.recognize_google(audio, language="vi-VN").lower(), "vi"
        except:
            return recognizer.recognize_google(audio, language="en-US").lower(), "en"
    except sr.UnknownValueError:
        return "", ""
    except Exception as e:
        print(f"Lỗi/Error: {e}")
        return "", ""

def process_feat2():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    wake_words = {
        "vi": ["này trợ lý", "ok máy tính", "xin chào python", "này python", "này", "xin chào"],
        "en": ["hey assistant", "ok computer", "hello python", "hey python", "hey", "hello"]
    }
    
    print("Đang lắng nghe từ khoá đánh thức... / Listening for wake words...")
    while True:
        text, lang = listen_and_recognize(recognizer, microphone)
        print(f"Đã nghe / Heard: {text}")
        
        if any(word in text for word in wake_words[lang]):
            if lang == "vi":
                text_to_speech("Tôi có thể giúp gì cho bạn?", lang)
            else:
                text_to_speech("How can I help you?", lang)
            
            command, lang = listen_and_recognize(recognizer, microphone)
            print(f"Lệnh / Command: {command}")
            
            if lang == "vi":
                if any(word in command for word in ["dừng", "không có gì", "thoát", "kết thúc"]):
                    text_to_speech("Tạm biệt!", lang)
                    break
                elif "ảnh" in command or "hình" in command:
                    text_to_speech("Đã chụp ảnh. Đang xử lý", lang)
                    result = process_feat1()
                    text_to_speech(result, lang)
                else:
                    text_to_speech("Tôi không hiểu lệnh đó. Vui lòng thử lại.", lang)
            else:
                if any(word in command for word in ["stop", "nothing", "exit", "quit"]):
                    text_to_speech("Goodbye!", lang)
                    break
                elif "photo" in command or "picture" in command:
                    text_to_speech("Picture taken. Processing", lang)
                    result = process_feat1()
                    text_to_speech(result, lang)
                else:
                    text_to_speech("I didn't understand that command. Please try again.", lang)

process_feat2()