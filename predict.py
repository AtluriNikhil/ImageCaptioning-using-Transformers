import torch
import threading
from transformers import BertTokenizer
from PIL import Image, ImageOps
import argparse
import streamlit as st
from streamlit.report_thread import add_report_ctx
import io
from models import caption
from datasets import coco, utils
from configuration import Config
import os
import base64

st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="Visualizer 2.0"
)

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

@st.cache
@torch.no_grad()
def evaluate(tokenizer):
    config = Config()
    model = torch.hub.load('AtluriNikhil/ImageCaptioning-using-Transformers', 'v3' , pretrained=True)
    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption

def result():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    output = evaluate(tokenizer)
    ans = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    return ans

@st.cache(ttl=3600, max_entries=10)
def load_output_image(img):
    
    if isinstance(img, str): 
        image = Image.open(img)
    else:
        img_bytes = img.read() 
        image = Image.open(io.BytesIO(img_bytes) ).convert("RGB")

    image = ImageOps.exif_transpose(image) 
    return image

if __name__ == '__main__':

    st.title("The Image Captioning Bot")
    st.text("")
    st.text("")
    st.info("Welcome! Please upload an image!")   

    args = { 'sunset' : 'data/girl.jpg' }
    
    img_upload  = st.file_uploader(label= 'Upload Image', type = ['png', 'jpg', 'jpeg','webp'])
    img_open = args['sunset'] if img_upload is None else img_upload
    image = load_output_image(img_open)


    st.sidebar.markdown('''
    # Intresting facts 
    If you are using Visualizer \n
    1. Website is supported on both PC :computer: and Mobile :phone:
    2. It works best with interaction of people with objects.
    3. Visualizer likes dogs :dog: , men, women and kids. Sorry catlovers.
    4. Profile pictures(Whatsapp) are \n good candidates!
    5. Very few animals work.

    If greater than/equal to two versions say
    you are an opposite gender, then you are more
    feminine looking and vice-versa.
    Upload a close-up to see! :wink:
    
    ''')
 
    st.sidebar.markdown('''
    # Upcoming features
    1. Make the model run faster
    2. Train the model on Full COCO dataset
    3. Make website more interactive

    Check the model details [here](https://github.com/sankalp1999/Image_Captioning)
    \n Liked it? Give a :star:  on GitHub
    
    ''')
    
    st.image(image,use_column_width=True,caption="Your image")

    if st.button('Generate captions!'):
        with st.spinner('Your Caption is being generated........'):
            image = coco.val_transform(image)
            image = image.unsqueeze(0)
            output = result()
            st.success(output.capitalize())
            st.info("Have fun by generating caption for different pictures!!")
        st.balloons()