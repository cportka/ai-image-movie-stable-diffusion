import argparse
import os
import subprocess
import torch
import re
import sys
import pickle
from diffusers import StableDiffusionPipeline
from transformers import CLIPImageProcessor
from pathlib import Path

class CustomStableDiffusionPipeline(StableDiffusionPipeline):
    def load_state_dict(self, state_dict):
        if "vae" in state_dict:
            self.vae.load_state_dict(state_dict["vae"])
        if "text_encoder" in state_dict:
            self.text_encoder.load_state_dict(state_dict["text_encoder"])
        if "unet" in state_dict:
            self.unet.load_state_dict(state_dict["unet"])
        if "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if "safety_checker" in state_dict:
            self.safety_checker.load_state_dict(state_dict["safety_checker"])
        if "feature_extractor" in state_dict:
            self.feature_extractor.load_state_dict(state_dict["feature_extractor"])

def download_and_save_model(model_id):
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    with open("sd-v1-4.pkl", "wb") as f:
        pickle.dump(pipe, f)
    return pipe

def load_local_model_from_ckpt(model_id, ckpt_path):
    if ckpt_path.endswith(".pkl"):
        print("Loading model from pickle file...")
        with open(ckpt_path, "rb") as f:
            pipe = pickle.load(f)
    else:
        print("Loading model from checkpoint file...")
        pipe = CustomStableDiffusionPipeline.from_pretrained(model_id)
        pipe.load_state_dict(torch.load(ckpt_path))
    return pipe

def load_local_model_from_pickle():
    with open("sd-v1-4.pkl", "rb") as f:
        pipe = pickle.load(f)
    return pipe

def get_local_model(model_id):
    ckpt_file = None
    for file in ["stable-diffusion-v1-4", "sd-v1-4.ckpt"]:
        if Path(file).is_file():
            ckpt_file = file
            break

    if ckpt_file:
        print(f"Loading model from local file '{ckpt_file}'")
        return load_local_model_from_ckpt(model_id, ckpt_file)
    elif Path("sd-v1-4.pkl").is_file():
        print("Loading model from local file 'sd-v1-4.pkl'")
        return load_local_model_from_pickle()
    else:
        print("Model files not found in the current directory.")
        download_choice = input("Would you like to download the model? (yes/no): ").lower()
        if download_choice == "yes":
            print("Downloading and saving model...")
            return download_and_save_model(model_id)
        else:
            print("Exiting the script.")
            sys.exit(0)

def generate_image(prompt, device, iteration, seed=None):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = get_local_model(model_id)
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
