import cv2
import os
import json as js
import requests
import pyaudio
import speech_recognition as sr
import wave
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound
import tempfile
import sys
import io
import sounddevice as sd
import soundfile as sf
import easyocr
from langdetect import detect
import google.generativeai as genai



sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
headers = {"Authorization": "Bearer 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"}
######################### MODEL USED ############################################
### OD
OD_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"

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


### Text extraction
genai.configure(api_key='AIzaSyAN7GhFw4Aaci7MupJnB8AtzT-JfiOATTA')

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 70,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

def get_ocr_text(filename):
    reader = easyocr.Reader(['vi', 'en'])
    result = reader.readtext(filename)
    text_in_frame = ""
    for image_result in result:
        if isinstance(image_result, (tuple, list)) and len(image_result) == 3:
            bbox, text, prob = image_result
        text_in_frame = text_in_frame + ' ' + text
    if text_in_frame == "":
        text_in_frame = " "
    response = model.generate_content(f"Hãy giúp tôi dự đoán đoạn text sau với những từ chưa được xác định do sai sót từ model OCR: {text_in_frame}")
    return response.text
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
    with open('count.txt', 'r') as fr:
        count = int(fr.readline())
    with open('count.txt', 'w') as fr:
        fr.write(str(count + 1))
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    image_path = os.path.join(folder, "captured_image.png")
    cv2.imwrite(image_path, frame)
    image_path = os.path.join('database\Keyframes', f'image_{count}.jpg')
    cv2.imwrite(image_path, frame)
    return image_path

def text_to_speech(text, lang="vi"):
    if not text:
        print("No text provided for text-to-speech.")
        return
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_filename = fp.name
    
    tts = gTTS(text=text, lang=lang)
    tts.save(temp_filename)
    
    try:
        data, fs = sf.read(temp_filename, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"Error playing sound: {e}")
    finally:
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

        paragraph = f"Tôi tin rằng trong hình có {objects_text}. "
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
            text = get_ocr_text(image_path)
            print(text)
            par = generate_image_description()
            text_to_speech(par)
            print(par)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        return par, text

def process_feat1_image(frame):
    try:
        image_path = save_image(frame)
        print(f"Image saved in {image_path}")
        get_obj_json(image_path)
        get_cap_json(image_path)
        text = get_ocr_text(image_path)
        par = generate_image_description()
        text_to_speech(par)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return par, text
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


def listen_and_recognize(recognizer, microphone, timeout=10):
    try:
        with microphone as source:
            print("Đang lắng nghe...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=None)
        try:
            print("Đang nhận diện...")
            text = recognizer.recognize_google(audio, language="vi-VN").lower()
            return text, "vi"
        except:
            text = recognizer.recognize_google(audio, language="en-US").lower()
            return text, "en"
    except sr.WaitTimeoutError:
        print("Hết thời gian chờ, không nhận được âm thanh.")
        return "", ""
    except sr.UnknownValueError:
        print("Không nhận diện được giọng nói.")
        return "", ""
    except Exception as e:
        print(f"Lỗi/Error: {e}")
        return "", ""

def process_feat2():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    wake_words = ["này trợ lý", "ok máy tính", "xin chào python", "này python", "này", "xin chào", "ê", "ê python",
                  "hey assistant", "ok computer", "hello python", "hey python", "hey", "hello"]
    
    while True:
        print("Đang lắng nghe từ khoá đánh thức... / Listening for wake words...")
        text, _ = listen_and_recognize(recognizer, microphone)
        print(f"Đã nghe / Heard: {text}")
        
        if any(word in text.lower() for word in wake_words):
            active = True
            while active:
                text_to_speech("Xin chào, tôi có thể giúp gì cho bạn?", "vi")
                
                command, _ = listen_and_recognize(recognizer, microphone, timeout=10)
                
                if not command:
                    text_to_speech("Không nhận được lệnh. Tôi sẽ chờ từ khóa đánh thức mới.", "vi")
                    active = False
                    continue
                
                print(f"Lệnh / Command: {command}")
                
                if any(word in command.lower() for word in ["dừng", "không có gì", "thoát", "kết thúc", "stop", "nothing", "exit", "quit"]):
                    text_to_speech("Tạm biệt! Tôi sẽ chờ từ khóa đánh thức mới.", "vi")
                    active = False
                elif any(word in command.lower() for word in ["ảnh", "hình", "photo", "picture"]):
                    text_to_speech("Đã chụp ảnh. Đang xử lý.", "vi")
                    process_feat1()
                else:
                    text_to_speech("Tôi không hiểu lệnh đó. Vui lòng thử lại.", "vi")

##########################################################################################################
#       FEATURE 3: GET IMAGE WITH CAM AND INFER, RETURN A VOICE DESCRIBING TRANSLATED TEXT IF EXIST      #
##########################################################################################################

def post_processing_text(filename):
    reader = easyocr.Reader(['vi', 'en', 'zh', 'es', 'ar', 'hi', 'ko', 'ja', 'th', 'lo', 'km'])
    result = reader.readtext(filename)
    text_in_frame = ""
    for image_result in result:
        if isinstance(image_result, (tuple, list)) and len(image_result) == 3:
            bbox, text, prob = image_result
        text_in_frame = text_in_frame + ' ' + text
    if text_in_frame == "":
        text_in_frame = " "
        text_to_speech("Đầu vào không thoả mãn. Xin hãy thử lại với đầu vào chứa văn bản")
    
    translator = GoogleTranslator(source='auto', target='vi')
    translated_text = translator.translate(text_in_frame)
    
    response = model.generate_content(f"{translated_text}")
    return response.text


def process_feat3():
    try:
        frame = capture_image()
        image_path = save_image(frame)
        print(f"Image saved in {image_path}")
        text = post_processing_text(image_path)
        text_to_speech(text)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return 

process_feat3()