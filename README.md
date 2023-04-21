# Text-to-Image-and-Video-Generator

This repository contains a Python script that generates images and videos based on a given text prompt using the Hugging Face Stable Diffusion v1-4 model.

## How it works

The script uses the [Stable Diffusion v1-4 Model Card](https://huggingface.co/CompVis/stable-diffusion-v1-4) to generate images based on the provided text prompt. It supports generating multiple images and creating a video from the generated images with a specified frame rate (fps).

## Requirements

1. Make sure you have Python 3 installed on your system.
2. Install the necessary Python packages:

pip install torch torchvision diffusers transformers

3. Download the [Stable Diffusion v1-4 Model Card](https://huggingface.co/CompVis/stable-diffusion-v1-4) and save it in the same directory as the `generate_image_and_video.py` script. The model file should be named `stable-diffusion-v1-4`.

## Usage
python generate_image_and_video.py "your text prompt here" --iterations N --fps M --seed S

- Replace `"your text prompt here"` with the desired text prompt for image generation.
- `N` is the number of images to generate (default is 1).
- `M` is the frames per second for the generated video (default is 24). This option is only relevant if `--iterations` is greater than 1.
- `S` is the random seed for image generation (default is None).

If no arguments are provided or an error occurs, a help message will be displayed with information on the proper usage of the script and acceptable parameters.

## Example

To generate four images with a 1 fps video using a seed of 42, run:

python generate_image_and_video.py --iterations 69 --seed 420 --fps 4 "a colorful star"

This will generate image files for each frameand an mp4 video file in the same directory as the script.
