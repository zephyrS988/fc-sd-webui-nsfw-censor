import os.path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers.utils import logging
from scripts.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from modules import scripts

logger = logging.get_logger(__name__)

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None




def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def check_safety(x_image, safety_checker_adj: float):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept, nsfw_count = safety_checker(
        images=x_image,
        clip_input=safety_checker_input.pixel_values,
        safety_checker_adj=safety_checker_adj,  # customize adjustment
    )

    return x_checked_image, has_nsfw_concept, nsfw_count


def censor_batch(x, safety_checker_adj: float):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, has_nsfw_concept, nsfw_count = check_safety(x_samples_ddim_numpy, safety_checker_adj)
    x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
    return x, nsfw_count


class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs['images']
        nsfw_count = 0

        enable_nsfw_check = True

        # Value range: [-0.5, 0.5], increasing this value will make the filter stronger
        adjustment = 0
        if enable_nsfw_check:
            images[:], nsfw_count = censor_batch(images, adjustment)[:]

        if p.extra_generation_params.get("NsfwCount") is None:
            p.extra_generation_params["NsfwCount"] = nsfw_count
        else:
            p.extra_generation_params["NsfwCount"] += nsfw_count
