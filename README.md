# Demo for Inverse Cooking with Streamlit

This project provides a **Streamlit-based UI** for the **Inverse Cooking** model, which generates recipes from food images. This is a simplified version of the original Inverse Cooking model, including only the necessary components for demonstration.

##  Features
- Upload an image of a dish 📸
- Get predicted ingredients 🥕
- Generate a recipe based on the image 🍲
- Simple & interactive UI with Streamlit 🎨

## 📂 Project Structure
```
📁 inverse_cooking_demo
 ├── data # (Ingredient vocabulary(ingr_vocab.pkl), Instruction vocabulary( instr_vocab.pkl), Pre-trained model checkpoint(modelbest))
 ├── demowith_streamlit.py   # Streamlit UI
 ├── model              # Recipe generation model (required functions)
 ├── utils               # Utility functions (metrics, loss functions, etc.)
 ├── assets                # Demo images
```

##  Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dathnet11/Demo-for-inversecooking.git
   cd Demo-for-inversecooking
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run demowith_streamlit.py
   ```

## Example Usage
1. Run the app and upload a food image.
2. The model will predict ingredients.
3. A recipe will be generated based on the predicted ingredients.




