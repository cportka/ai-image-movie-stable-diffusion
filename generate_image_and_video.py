import argparse
import os
import subprocess
import torch
import re
from diffusers import StableDiffusionPipeline
from transformers import CLIPImageProcessor

def generate_image(prompt, device, iteration, seed=None):
    model_id = "CompVis/stable-diffusion-v1-4"

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    if seed is not None:
        torch.manual_seed(seed + iteration)
        torch.cuda.manual_seed_all(seed + iteration)

    image = pipe(prompt).images[0]

    sanitized_prompt = re.sub(r'\W+', '_', prompt)
    filename = f"{sanitized_prompt}_generated_image_{iteration}.png"
    image.save(filename)
    return filename

def create_video(filenames, prompt, fps=1):
    sanitized_prompt = re.sub(r'\W+', '_', prompt)
    output_name = f"{sanitized_prompt}_generated_video.mp4"

    with open("file_list.txt", "w") as f:
        for filename in filenames:
            f.write(f"file '{filename}'\nduration {1/fps}\n")
        f.write(f"file '{filenames[-1]}'\n")  # Add the last image without duration to fix end time

    ffmpeg_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "file_list.txt", "-c:v", "libx264", "-profile:v", "high", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(fps), output_name]

    subprocess.run(ffmpeg_cmd)

    os.remove("file_list.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from text prompt.")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for image generation.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for generated video.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = args.prompt
    iterations = args.iterations
    seed = args.seed

    generated_filenames = []
    for i in range(iterations):
        generated_filename = generate_image(prompt, device, i, seed)
        generated_filenames.append(generated_filename)

    if iterations > 1:
        create_video(generated_filenames, prompt, args.fps)
