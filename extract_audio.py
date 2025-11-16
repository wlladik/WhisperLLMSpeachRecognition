from moviepy import VideoFileClip
import os

def extract_audio(video_path, output_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path)

extract_audio("data/video/IMG_4986.MP4", "data/audio/audio.wav")