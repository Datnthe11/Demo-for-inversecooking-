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
from model import get_model

# Thiết bị tính toán (CPU hoặc GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mô hình từ hàm get_model trong file model.py
def load_model(model_path, args, ingr_vocab_size, instrs_vocab_size, device):
    model = get_model(args, ingr_vocab_size, instrs_vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Tiền xử lý ảnh
def preprocess_image(image, model_name="resnet101"):
    image_size = 299 if "inception" in model_name else 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Giải mã đầu ra
def decode_output(output, ingr_vocab, instr_vocab):
    ingr_ids = output["ingr_ids"][0].tolist()  # Chuyển tensor thành list
    recipe_ids = output["recipe_ids"][0].tolist()

    # Loại bỏ các token "<PAD>"
    ingredients = [ingr_vocab.get(idx, f"<UNK-{idx}>") for idx in ingr_ids if ingr_vocab.get(idx, "") not in ["<PAD>", "<pad>"]]
    
    # Loại bỏ các token "<eoi>" và "<end>"
    recipe_steps = [instr_vocab.get(idx, f"<UNK-{idx}>") for idx in recipe_ids if instr_vocab.get(idx, "") not in ["<eoi>", "<end>"]]

    return {
        "ingredients": ", ".join(ingredients),
        "instructions": " ".join(recipe_steps)
    }




# Dự đoán công thức
def predict_recipe(image, model, ingr_vocab, instr_vocab):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model.sample(input_tensor)
    return decode_output(output, ingr_vocab, instr_vocab)

# Load mô hình và từ điển
# Thay thế từ đường link trong máy bạn
parser = get_parser()  
args = parser.parse_args()

model_path = r"C:\Users\Dat Nguyen\Downloads\inversecooking\data\modelbest.ckpt"
ingr_vocab_path = r"C:\Users\Dat Nguyen\Downloads\inversecooking\data\ingr_vocab.pkl"
instr_vocab_path = r"C:\Users\Dat Nguyen\Downloads\\inversecooking\data\instr_vocab.pkl"

with open(ingr_vocab_path, "rb") as f:
    ingr_vocab = pickle.load(f)

with open(instr_vocab_path, "rb") as f:
    instr_vocab = pickle.load(f)
if isinstance(ingr_vocab, list):
    ingr_vocab = {i: word for i, word in enumerate(ingr_vocab)}



model = load_model(model_path, args, len(ingr_vocab), len(instr_vocab), device)



# Giao diện Streamlit
st.title("Inverse Cooking - Dự đoán công thức món ăn từ ảnh")
st.write("Tải ảnh món ăn của bạn lên để nhận công thức.")

# Tải ảnh lên
uploaded_file = st.file_uploader("Chọn ảnh món ăn", type=["jpg", "png", "jpeg"], key="uploader_1")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh món ăn", use_column_width=True)

    if st.button("Dự đoán công thức"):
        st.write("🔍 Đang chạy mô hình...")
        
        # Gọi hàm dự đoán công thức
        recipe = predict_recipe(image, model, ingr_vocab, instr_vocab)
        print("Recipe output:", recipe)


        st.write("✅ Mô hình chạy xong!")

        # Hiển thị kết quả
        st.write("### Công thức gợi ý:")
        st.write(f"**Nguyên liệu:** {recipe['ingredients']}")
        st.write(f"**Công thức:** {recipe['instructions']}")