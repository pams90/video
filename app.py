# Import required libraries
import os
import torch
import streamlit as st  # Streamlit for the web interface
from diffusers import DiffusionPipeline  # Text-to-video model
from ffmpeg import input as ffmpeg_input  # For video processing

# Set page title and description
st.set_page_config(
    page_title="AI Video Generator",
    page_icon="🎥",
    layout="centered",
)

# Title and description
st.title("AI Video Generator")
st.write("Create short videos from text prompts using AI. Example: 'A cat dancing in the rain'")

# Function to load the text-to-video model
def load_model():
    """
    Load the text-to-video model from Hugging Face.
    Uses half-precision (fp16) for better performance on GPUs.
    """
    return DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",  # Model name
        torch_dtype=torch.float16,  # Use half-precision for faster inference
        variant="fp16",  # Load the fp16 version of the model
        device_map="auto"  # Automatically handle GPU/CPU allocation
    ).to("cuda")  # Move the model to GPU

# Function to generate video from text
def generate_video(prompt):
    """
    Generate a video from a text prompt.
    Args:
        prompt (str): The text description of the video.
    Returns:
        str: Path to the generated video file.
    """
    try:
        # Create a temporary directory for storing frames
        os.makedirs("temp", exist_ok=True)
        
        # Load the model (cached after first run)
        pipe = load_model()
        
        # Generate video frames from the text prompt
        st.write(f"Generating video for: {prompt}")
        video_frames = pipe(prompt, num_inference_steps=25).frames
        
        # Save each frame as an image
        for i, frame in enumerate(video_frames):
            frame.save(f"temp/frame_{i:04d}.png")
        
        # Use FFmpeg to combine frames into a video
        (
            ffmpeg_input("temp/frame_%04d.png", framerate=8)  # Input frames at 8 FPS
            .output("output.mp4", vcodec="libx264", pix_fmt="yuv420p")  # Output as MP4
            .run()  # Execute FFmpeg command
        )
        
        # Clean up temporary frame files
        for frame in os.listdir("temp"):
            os.remove(f"temp/{frame}")
        
        # Return the path to the generated video
        return "output.mp4"
    
    except Exception as e:
        # Handle errors gracefully
        st.error(f"Error generating video: {e}")
        return None

# Streamlit UI
prompt = st.text_input("Enter your prompt", placeholder="A cat dancing in the rain")

if st.button("Generate Video"):
    if prompt:
        with st.spinner("Generating video..."):
            video_path = generate_video(prompt)
            if video_path:
                st.success("Video generated successfully!")
                st.video(video_path)
    else:
        st.warning("Please enter a prompt.")