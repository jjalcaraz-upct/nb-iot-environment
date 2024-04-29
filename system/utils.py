#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary function for movie generation

Created on Jan 13, 2022

@author: juanjosealcaraz

"""

import subprocess
from pathlib import Path

# auxiliary function
def ffmpeg_movie(frames_dir: Path, movie_dir: Path, framerate: int = 15) -> None:
    image_pattern = frames_dir / '*png'
    print(image_pattern)
    savepath = movie_dir.with_suffix('.mp4')
    print(savepath)

    # It's a lengthy command, but there's no need to grok it
    command = f"""
        ffmpeg -y -framerate {framerate} -f image2 -pattern_type glob \
        -i '{image_pattern}' -c:v libx264 -r 30 -profile:v high -crf 20 \
        -pix_fmt yuv420p {savepath}
    """
    subprocess.call(command, shell=True)

def generate_movie(movie_name = 'movie'):
    frames_dir = Path('./frames')
    movie_dir = Path(f'./movies/{movie_name}')
    ffmpeg_movie(frames_dir, movie_dir)

def find_next_integer(lst, n):
    '''
    Finds the smallest integer in the list 'lst' that is greater or equal than integer 'n'
    '''
    greater_integers = [i for i in lst if i >= n]
    if greater_integers:
        return min(greater_integers)
    else:
        return max(lst) # for safety
    
def find_next_integer_index(lst, n):
    """
    Finds the smallest integer in the list 'lst' that is greater or equal than integer 'n' and its position
    """
    greater_integers = [(i, j) for j, i in enumerate(lst) if i >= n]
    if greater_integers:
        return min(greater_integers)
    else:
        return max(lst), len(lst)-1
    
def read_flag_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            flag = file.read().strip()
            return flag.lower() == 'true'
    except FileNotFoundError:
        return False