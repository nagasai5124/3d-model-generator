
import streamlit as st
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image
from shap_e.util.notebooks import decode_latent_mesh
import io
import time
import tempfile

st.title("create 3d model")
option = st.selectbox(
    "select the model",
    ("image-to-3d", "text-to-3d")
)

# Create a placeholder
placeholder = st.empty()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if option == "text-to-3d":
    prompt = st.text_input("Enter the text")
    if st.button("Create 3D Model"):
        # create 3d model using text
        import os
        import shutil

        # This is the default cache directory for CLIP models
        clip_cache_dir = os.path.expanduser("~/.cache/clip")

        # Remove the entire CLIP cache directory
        if os.path.exists(clip_cache_dir):
            shutil.rmtree(clip_cache_dir)
        placeholder.success("loading model.....")
        # Re-run the model loading code
        xm = load_model('transmitter', device=device)
        text_model = load_model('text300M', device=device)  # This will now download a fresh copy
        diffusion = diffusion_from_config(load_config('diffusion'))
        time.sleep(5)
        placeholder.empty()
        placeholder.success("generating 3d models...") 
        # generate 3d model
        batch_size = 4
        guidance_scale = 50.0
        latents = sample_latents(
            batch_size=batch_size,
            model=text_model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        render_mode = 'nerf' # you can change this to 'stf'
        size = 64 # this is the size of the renders; higher values take longer to render.

        cameras = create_pan_cameras(size, device)
        def show_gif_in_streamlit(images, duration=100, target_size=(64, 64)):
          # images: list of PIL.Image
          if not images:
              st.warning("No images to display.")
              return

          # Resize each image
          resized_images = [img.resize(target_size) for img in images]

          # Save images as GIF in memory
          gif_buffer = io.BytesIO()
          resized_images[0].save(
              gif_buffer,
              format='GIF',
              save_all=True,
              append_images=resized_images[1:],
              duration=duration,
              loop=0
          )
          gif_buffer.seek(0)

          # Display resized GIF
          st.image(gif_buffer, caption="3D Render (Animated)")

        for i, latent in enumerate(latents):
            images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            time.sleep(5)
            placeholder.empty()
            #Visualization
            placeholder.success(f"{prompt} 3D model")
            show_gif_in_streamlit(images)
            #mesh obj
            t = decode_latent_mesh(xm, latent).tri_mesh()
            obj_text = io.StringIO()
            t.write_obj(obj_text)
            obj_bytes = io.BytesIO(obj_text.getvalue().encode("utf-8"))  # Convert text to bytes
            obj_bytes.seek(0)  
            st.download_button(
              label=f"Download {prompt} Mesh {i} (.obj)",
              data=obj_bytes.getvalue(),
              file_name=f"{prompt}_{i}.obj",
              mime="text/plain"
          )
else:
  uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
  if st.button("Create 3D Model"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_image.read())
        tmp_path = tmp.name
    # create 3d model using image

    import os
    import shutil

    # This is the default cache directory for CLIP models
    clip_cache_dir = os.path.expanduser("~/.cache/clip")

    # Remove the entire CLIP cache directory
    if os.path.exists(clip_cache_dir):
        shutil.rmtree(clip_cache_dir)
    placeholder.success("loading model.....")
    # Re-run the model loading code
    xm = load_model('transmitter', device=device)
    image_model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    time.sleep(5)
    placeholder.empty()
    placeholder.success("generating 3d models...") 
    batch_size = 4
    guidance_scale = 3.0

    # To get the best result, you should remove the background and show only the object of interest to the model.
    image = load_image(tmp_path)

    latents = sample_latents(
        batch_size=batch_size,
        model=image_model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    render_mode = 'nerf' # you can change this to 'stf'
    size = 64 # this is the size of the renders; higher values take longer to render.

    cameras = create_pan_cameras(size, device)
    def show_gif_in_streamlit(images, duration=100, target_size=(64,64)):
    # images: list of PIL.Image
      if not images:
          st.warning("No images to display.")
          return

      # Resize each image
      resized_images = [img.resize(target_size) for img in images]

      # Save images as GIF in memory
      gif_buffer = io.BytesIO()
      resized_images[0].save(
          gif_buffer,
          format='GIF',
          save_all=True,
          append_images=resized_images[1:],
          duration=duration,
          loop=0
      )
      gif_buffer.seek(0)

      # Display resized GIF
      st.image(gif_buffer, caption="3D Render (Animated)")

    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        time.sleep(5)
        placeholder.empty()
        #Visualization
        placeholder.success(f"3D model")
        show_gif_in_streamlit(images)
        #mesh obj
        t = decode_latent_mesh(xm, latent).tri_mesh()
        obj_text = io.StringIO()
        t.write_obj(obj_text)
        obj_bytes = io.BytesIO(obj_text.getvalue().encode("utf-8"))  # Convert text to bytes
        obj_bytes.seek(0)  
        st.download_button(
          label=f"Download  Mesh {i} (.obj)",
          data=obj_bytes.getvalue(),
          file_name=f"mesh_{i}.obj",
          mime="text/plain"
      )

