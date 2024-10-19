import streamlit as st
from diffusers import DiffusionPipeline
import torch

st.title("Text generation image application")


#load DiffusionPipeline model
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.to("cpu")
    return pipe


pipe = load_model()
prompt = st.text_input("Please enter the image description you want to generate:")
if st.button("Generate Images"):
    if prompt:
        with st.spinner("Generating..."):
            image = pipe(prompt).images[0]

        st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter description text!")
