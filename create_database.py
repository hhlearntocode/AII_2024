import numpy as np
import faiss
import glob
import json
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class File4Faiss:
  def __init__(self, root_database: str):
    self.root_database = root_database

  def write_json_file(self, json_path: str, option='full'):
    count = 0
    self.infos = []
    des_path = os.path.join(json_path, "keyframes_id.json")
    image_paths = sorted(glob.glob(f'{self.root_database}\KeyFrames\*.jpg'))
    id2img_fps = {}
    for image_path in image_paths:
        id2img_fps[str(count)] = image_path
        count += 1
    with open(des_path, 'w') as f:
      f.write(json.dumps(id2img_fps))
    print(f'Saved {des_path}')
    print(f"Number of Index: {count}")

  def load_json_file(self, json_path: str):
    with open(json_path, 'r') as f:
      js = json.loads(f.read())

    return {int(k):v for k,v in js.items()}
  
  def write_bin_file(self, bin_path: str, json_path: str, method='L2', feature_shape=512):
    keyframes_dir = 'database/Keyframes'
    # Load CLIP model và processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Danh sách chứa các vector embedding
    embeddings = []

    # Duyệt qua các ảnh trong thư mục
    for filename in os.listdir(keyframes_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Mở và xử lý ảnh
            image_path = os.path.join(keyframes_dir, filename)
            image = Image.open(image_path)
            
            # Tiền xử lý và tạo embedding
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = model.get_image_features(**inputs).cpu().numpy()
            
            embeddings.append(embedding)

    # Chuyển các embeddings thành numpy array
    embeddings = np.vstack(embeddings)

    # Tạo FAISS index và thêm các embeddings vào index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Lưu FAISS index
    faiss.write_index(index, bin_path + '\\faiss_cosine.bin')
    print(f"Đã lưu {len(embeddings)} ảnh vào FAISS index.")
#create_file = File4Faiss('database')
#create_file.write_json_file(json_path='database')
#create_file.write_bin_file(bin_path='database', json_path='database\keyframes_id.json', method='cosine')