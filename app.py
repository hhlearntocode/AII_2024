import streamlit as st
from PIL import Image
import numpy as np
from function import process_feat1, process_feat1_image
from create_database import File4Faiss
from retrieval_func import MyFaiss
import os
import cv2
st.set_page_config(layout="wide")
st.title("AI-Powered Smart Glasses - Real-Time Hazard Warning and Information Accessibility for the Visually Impaired")

###########################################
bin_file = 'database/faiss_cosine.bin'
json_path = 'database/keyframes_id.json'
cosine_faiss = MyFaiss('', bin_file, json_path)
###########################################
database = st.sidebar.button('Update database')
Capture_image = st.sidebar.button('Capture_image')
Assistant = st.sidebar.button('Assistant')
Retrieval_input = st.sidebar.text_input('Moi nhap truy van: ')
Retrieval = st.sidebar.button('Retrieval')
image_uploader = st.sidebar.file_uploader("Chọn một hình ảnh", type=["jpg", "jpeg", "png"])
###########################################

if Capture_image:
    st.write('Image is being taken')
    describing, text = process_feat1()
    st.write(describing)
    st.write(text)
    
    
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
    # create_file = File4Faiss('database')
    # create_file.write_json_file(json_path='database')
    # create_file.write_bin_file(bin_path='database', json_path='database\keyframes_id.json', method='cosine')
    st.write('Successfully !!!')

if image_uploader is not None:
    image = Image.open(image_uploader)
    st.image(image, caption='Ảnh đã tải lên.', use_column_width=True)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img_upload_process = st.button('Process')
    if img_upload_process:
            describing, text = process_feat1_image(frame)
            st.write(describing)
            st.write(text)