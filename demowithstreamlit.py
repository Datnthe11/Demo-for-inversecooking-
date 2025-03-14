import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "src")

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import module
from utils.metrics import softIoU, MaskedCrossEntropyCriterion
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from args import get_parser
from src.model import get_model
from src.utils.output_utils import prepare_output

# Xác định thiết bị (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Định nghĩa hàm load mô hình
def load_model(model_path, args, ingr_vocab_size, instr_vocab_size, device):
    model = get_model(args, ingr_vocab_size, instr_vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Hàm tiền xử lý ảnh
def preprocess_image(image, model_name="resnet101"):
    image_size = 299 if "inception" in model_name else 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Hàm dự đoán công thức
def predict_recipe(image, model, ingr_vocab, instr_vocab):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model.sample(input_tensor)
        print(output)
        if not output or "recipe_ids" not in output or len(output["recipe_ids"]) == 0:
            raise ValueError("Mô hình không sinh ra bất kỳ công thức nào. Kiểm tra lại input và model.")
        
        ingr_ids = output['ingr_ids'].cpu().numpy()
        recipe_ids = output['recipe_ids'].cpu().numpy()
        
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingr_vocab, instr_vocab)
        
        if not valid['is_valid']:
            raise ValueError(f"Công thức không hợp lệ. Lý do: {valid['reason']}")
        
        return outs

# Load mô hình và từ điển
@st.cache_resource()
def load_resources():
    parser = get_parser()
    args = parser.parse_args([])  # Tránh lỗi khi chạy trên Streamlit
    
    model_path = r"\inversecooking\data\modelbest.ckpt"
    ingr_vocab_path = r"\inversecooking\data\ingr_vocab.pkl"
    instr_vocab_path = r"\inversecooking\data\instr_vocab.pkl"
    
    with open(ingr_vocab_path, "rb") as f:
        ingr_vocab = pickle.load(f)

    with open(instr_vocab_path, "rb") as f:
        instr_vocab = pickle.load(f)
    
    if isinstance(ingr_vocab, list):
        ingr_vocab_list = ingr_vocab
        ingr_vocab = {i: word for i, word in enumerate(ingr_vocab_list)}
    else:
      ingr_vocab_list = list(ingr_vocab.values())

    model = load_model(model_path, args, len(ingr_vocab), len(instr_vocab), device)
    
    return model, ingr_vocab_list, instr_vocab

# Tải tài nguyên
model, ingr_vocab, instr_vocab = load_resources()

# Giao diện Streamlit
st.title("🍽️ Inverse Cooking - Dự đoán công thức món ăn từ ảnh")
st.write("Tải ảnh món ăn của bạn lên để nhận công thức gợi ý!")

# Tải ảnh lên
uploaded_file = st.file_uploader("📸 Chọn ảnh món ăn", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh món ăn", use_column_width=True)
    
    if st.button("🎯 Dự đoán công thức"):
        st.write("⏳ Đang chạy mô hình...")
        
        try:
            recipe = predict_recipe(image, model, ingr_vocab, instr_vocab)
            st.write("✅ **Mô hình đã hoàn thành!**")

            st.subheader("🥕 Nguyên liệu:")
            for ingredient in recipe["ingrs"]:
                st.write(f"- {ingredient}")

            st.subheader("📜 Công thức:")
            for step in recipe["recipe"]:
              st.write(f"- {step}")

        except Exception as e:
            st.error(f"Lỗi: {e}")
