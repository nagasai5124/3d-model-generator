# 3D Model Generation from Text/Image 🖼️➡️🧊

A Streamlit web application that generates 3D models (.obj) from either:
- Text prompts ("A small toy car")
- Images (PNG/JPG with object focus)

Built with OpenAI's Shap-E and PyTorch. 

https://github.com/user-attachments/assets/3fa789d1-7854-4bc0-938e-e66569705f58


![text-to-3d ipynb - Colab - Google Chrome 5_4_2025 6_03_17 PM](https://github.com/user-attachments/assets/8144c103-37c2-4a31-aac2-d248feb5bfbc)

## Features ✨
- Text-to-3D generation
- Image-to-3D conversion
- 3D model preview (animated GIF)
- Direct .obj file download
- GPU acceleration support

## Workflow
1.Image Processing: Uses rembg for background removal.

2. 3D Generation: Uses Shap-E for image/text-to-3D conversion.

3. Output: Saves as .obj and displays a 3D scatter plot.

Installation 💻
Prerequisites
Python 3.8-3.12

Git (for Shap-E installation)

CUDA (optional but recommended)

# Steps
1 step:
Clone repository:
!git clone https://github.com/openai/shap-e.git
%cd shap-e
!pip install -e .

2 step:
Create virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3 step:
Install dependencies:
pip install -r requirements.txt

4.step
Usage 🚀
Start the app:
streamlit run app.py

Choose mode:

Text-to-3D: Enter prompt and click "Create 3D Model"

Image-to-3D: Upload image and click "Create 3D Model"

Wait for processing (2-5 minutes depending on hardware)

Preview the 3D model and download .obj file
