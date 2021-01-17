import subprocess
import ffmpeg
import os.path
my_images_dir = 'D:/yolo/process_images'
'''convert image sequence to movie'''

input = 'D:/yolo/process_images/%04d.jpg' # all the picturs in the folder with 4 numbers
output = r"D:/yolo/dsp_video.avi"
frame_rate = 24
cmd = f'ffmpeg -framerate {frame_rate} -i "{input}" "{output}"'

subprocess.check_output(cmd,shell=True)
