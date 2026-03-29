from google.colab import drive
drive.mount('/content/drive')

!pip uninstall -y googletrans
!pip install googletrans==4.0.0-rc1
!pip install --upgrade diffusers transformers -q

from googletrans import Translator
from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2

def get_translation(text, dest_lang):
  translator = Translator()
  translated_text = translator.translate(text, dest=dest_lang)
  return translated_text.text

class CFG:
  device = "cuda"
  seed = 42

  try:
    generator = torch.Generator(device).manual_seed(seed)
  except Exception as e:
    print(f"Error: {e}")
    device = "cpu"
    generator = torch.Generator(device).manual_seed(seed)

  image_gen_steps = 35
  image_gen_model_id = "runwayml/stable-diffusion-v1-5"
  image_gen_size = (900,900)
  image_gen_guidance_scale = 9
  prompt_gen_model_id = "gpt-omni/mini-omni"
  prompt_dataset_size = 6
  prompt_max_lenghth = 12
  
  from diffusers import StableDiffusionPipeline
import torch

model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token="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")
pipe = pipe.to("cuda")

from huggingface_hub import notebook_login
from google.colab import userdata

# This will prompt you to enter your token if it's not already set up
# or if you prefer to log in directly.
# Alternatively, if you've saved it in Colab Secrets as 'HF_TOKEN':
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception as e:
    print(f"Could not retrieve HF_TOKEN from Colab Secrets: {e}")
    HF_TOKEN = None

if HF_TOKEN is None:
    print("Hugging Face token not found in Colab Secrets. Please set it up as 'HF_TOKEN'.")
    # Fallback to notebook_login if token not found in secrets
    notebook_login()
else:
    print("Hugging Face token loaded from Colab Secrets.")
    
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)

image_gen_model = image_gen_model.to(CFG.device)

import torch
from diffusers import StableDiffusionPipeline
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype = torch.float16,
    revision='fp16', use_auth_token= '', guidance_scale=9
)

image_gen_model = image_gen_model.to(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)

#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

def generate_image(prompt, model):
  image = model(
      prompt, num_inference_steps=CFG.image_gen_steps,
      generator = CFG.generator,
      guidance_scale=CFG.image_gen_guidance_scale
  ).images[0]


  image = image.resize(CFG.image_gen_size)
  return image

translation = get_translation("भारतीय उत्सव गणेश विसर्जन","en")
generate_image(translation, image_gen_model)


