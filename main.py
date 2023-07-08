from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from PIL import Image
import base64
from io import BytesIO
import qrcode
import torch
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
)
from diffusers.utils import load_image
import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


# Pydantic model for request body
class ImageRequest(BaseModel):
    qrcode_data: str
    prompt: str
    negative_prompt: str
    guidance_scale: float
    controlnet_conditioning_scale: float
    generator_seed: int
    strength: float


app = FastAPI()


@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    # Create a QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(request.qrcode_data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    basewidth = 768
    wpercent = basewidth / float(img.size[0])
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.LANCZOS)

    # Initialize API key and generate initial image
    openai.api_key = "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"
    response = openai.Image.create(prompt=request.prompt, n=1, size="1024x1024")
    image_url = response.data[0].url

    # Initialize the control net model and pipeline.
    controlnet = ControlNetModel.from_pretrained(
        "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
    )

    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    def resize_for_condition_image(input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img

    init_image = load_image(image_url)
    condition_image = img
    init_image = resize_for_condition_image(init_image, 768)
    generator = torch.manual_seed(request.generator_seed)

    image = pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        image=init_image,
        control_image=condition_image,
        width=768,
        height=768,
        guidance_scale=request.guidance_scale,
        controlnet_conditioning_scale=request.controlnet_conditioning_scale,
        generator=generator,
        strength=request.strength,
        num_inference_steps=150,
    )

    pil_image = image.images[0]

    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content={"image": image_base64})
