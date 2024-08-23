import streamlit as st
from PIL import Image
import numpy as np
from function import process_feat1, process_feat1_image, text_to_speech, process_feat3, process_feat4
from create_database import File4Faiss
from retrieval_func import MyFaiss
import os
import cv2 
from face_recognition import face_recognition
st.set_page_config(layout="wide")
st.title("AI-Powered Smart Glasses - Real-Time Hazard Warning and Information Accessibility for the Visually Impaired")

###########################################
bin_file = 'database/faiss_cosine.bin'
json_path = 'database/keyframes_id.json'
cosine_faiss = MyFaiss('', bin_file, json_path)
Face = face_recognition()
###########################################
database = st.sidebar.button('Update database')
Capture_image = st.sidebar.button('Capture_image')
Assistant = st.sidebar.button('Assistant')
ocr = st.sidebar.button('Extract text')
Retrieval_input = st.sidebar.text_input('Moi nhap truy van: ')
Retrieval = st.sidebar.button('Retrieval')
image_uploader = st.sidebar.file_uploader("Chọn một hình ảnh", type=["jpg", "jpeg", "png"])
find_func = st.sidebar.text_input('Moi nhap do vat can tim kiem: ')
find_button = st.sidebar.button('Find')
###########################################

if Capture_image:
    st.write('Image is being taken')
    describing = process_feat1()
    image_path = "image/captured_image.png"
    image = Image.open(image_path)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
    label, distance = Face.find_person(image_path)
    if distance < 0.5:
        st.write(f"Dự đoán: {label} với khoảng cách {distance:.4f}")
        text_to_speech(f"Dự đoán: {label} với khoảng cách {distance:.4f}")
    else:
        # name =  st.text_input('Nhap ten moi: ')
        # if name: 
        #     Face.create_new_face(name, frame)
        st.write('Unknown!!!')
    st.write(describing)
    
elif Assistant:
    st.write('Assistant is listening.........')
    #process_feat2()
    
elif Retrieval and Retrieval_input:
    st.write(Retrieval_input)
    image_paths = cosine_faiss.text_search(Retrieval_input, k = 9)
    num_cols = 4  # Number of columns for image display
    cols = st.columns(num_cols)
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        with cols[i % num_cols]:
            st.image(img, caption=f"{os.path.basename(image_path)}", use_column_width=True)
        

elif database:
    create_file = File4Faiss('database')
    create_file.write_json_file(json_path='database')
    create_file.write_bin_file(bin_path='database', json_path='database\keyframes_id.json', method='cosine')
    st.write('Successfully !!!')

elif image_uploader is not None:
    image = Image.open(image_uploader)
    st.image(image, caption='Ảnh đã tải lên.', use_column_width=True)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
    describing, text = process_feat1_image(frame)
    image_path = "image/captured_image.png"
    label, distance = Face.find_person(image_path)
    if distance < 0.3:
        st.write(f"Dự đoán: {label} với khoảng cách {distance:.4f}")
        text_to_speech(f"Dự đoán: {label} với khoảng cách {distance:.4f}")
    else:
        name =  st.text_input('Nhap ten moi: ')
        if name: 
            Face.create_new_face(name, frame)
    st.write(describing)
    st.write(text)


elif ocr:
    text = process_feat3()
    st.write(text)

elif find_func and find_button:
    image_path = process_feat4()
    similarity = cosine_faiss.image_warning(find_func, image_path)
    st.write(similarity)
        
