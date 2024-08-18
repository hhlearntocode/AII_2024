import streamlit as st
from PIL import Image
import numpy as np
from function import process_feat1
from retrieval_func import MyFaiss
import os
st.set_page_config(layout="wide")
st.title("AI-Powered Smart Glasses - Real-Time Hazard Warning and Information Accessibility for the Visually Impaired")

###########################################
bin_file = 'database/faiss_cosine.bin'
json_path = 'database/keyframes_id.json'
cosine_faiss = MyFaiss('', bin_file, json_path)
###########################################
Capture_image = st.sidebar.button('Capture_image')
Assistant = st.sidebar.button('Assistant')
Retrieval = st.sidebar.button('Retrieval')
Retrieval_input = st.sidebar.text_input('Moi nhap truy van: ')
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
        
