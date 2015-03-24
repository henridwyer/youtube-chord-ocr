#!/usr/bin/python
# coding: latin-1

import numpy as np
from numba import jit
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image#, ImageOps, ImageFilter
from collections import OrderedDict, Counter
import subprocess as sp
import cv2
import os
import wave
import json

################################################
############## Global Parameters ###############
################################################

OCR_BIN = "ocrad"
# the ocr program uses these files for input/output
IMG_FILENAME = "ocr.PPM"
OUTPUT_FILENAME = "ocr.orf"
OUTPUT_DIR = 'output'

SKIP_OCR_THRESH = 0.01
TOP_TRUNC = 570
COLOR_TRUNC = 250
C_SUM_MAX = 2
C_SUM_MIN = 5
NOTE_REGEX = re.compile(ur"(([A-G]|[ABEG]♭|[CF]♯?)(maj|min|[Mm+°])?6?(aug|d[io]m|ø)?7?)")
QUOTE_REGEX = re.compile(r"'(.)'")
BOX_REGEX = re.compile(r"(\d+)\ *(\d+)\ *(\d+)\ *(\d+)")
DEBUG = False
MAX_CHORDS = 10

# seconds
MIN_CHORD_MERGE_LENGTH = 0.4
MIN_CHORD_LENGTH = 0.3
DELAY = 0.1 # delay audio
# remove the final part of the video because it just contains the logo
REMOVE_FINAL = 25

# Global variable to store previous frame
previous_frame_global = None


################################################
####### Image Transformation Functions #########
################################################

def binarize(image):
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)[1]

def img_pre_process(image_arr):
    trunc_img = trunc_image(image_arr, TOP_TRUNC)
    bin_img = binarize(trunc_img)
    return bin_img

# OpenCV stores image colors as BGR instead of what matplotlib expects: RGB
# This function allows easy plotting of OpenCV images
def show_img(image, swap_colors = True):
    if len(image.shape) == 3 and image.shape[2] == 3:
        if swap_colors:
            plt.imshow(image[:,:,[2,1,0]])
        else:
            plt.imshow(image[:,:,:])
    else:
        plt.imshow(image, cmap = cm.Greys_r)

def trunc_image(image, *trunc):
    if len(trunc) == 4: # it's a box
        box = trunc
        if len(image.shape) == 2:
            return image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        elif len(image.shape) == 3:
            return image[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]
    elif len(trunc) == 1: # just trunc the top
        if len(image.shape) == 2:
            return image[trunc[0]:,:]
        elif len(image.shape) == 3:
            return image[trunc[0]:,:,:]
    elif len(trunc) == 2: #trunc top and left
        if len(image.shape) == 2:
            return image[trunc[0]:,trunc[1]:]
        elif len(image.shape) == 3:
            return image[trunc[0]:,trunc[1]:,:]
    raise "Could not truncate imange: Invalid arguments."



################################################
############### OCR Functions ##################
################################################

# Find the active chord for each frame
def vidcap_to_frame_chords(vidcap, video_fps, nb_frames = -1):
    if nb_frames == -1:
        nb_frames = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frame_chords = []
    final_frame = int(nb_frames - REMOVE_FINAL * video_fps)
    for i in range(final_frame - 1):
        success,image = vidcap.read()
        frame_chords.append(get_active_chord(image))
    return frame_chords

# Find the active chord for a given frame
def get_active_chord(image):
    pre_proc = img_pre_process(image)
    image_ocr(pre_proc)
    chords, chord_boxes = ocr_output_to_coords()

    if not chords or len(chords) > MAX_CHORDS: #no chords read
        return False

    ratios = black_non_white_ratio(chord_boxes, pre_proc)
    active_chord_index = np.argmax(ratios)
    # if one chord is less black than the others, and the other are close enough to black
    # then the active chord exists and is the the one corresponding to the maximum ratio
    if ratios[active_chord_index] > C_SUM_MIN and all([a < C_SUM_MAX for i,a in enumerate(ratios) if i != active_chord_index]):
        return chords[active_chord_index]
    else:
        return False

def image_ocr(image_arr):
    global previous_frame_global
    if previous_frame_global is not None:
        assert image_arr.shape == previous_frame_global.shape, "Frames have different dimensions." #should never happen
        total_values = image_arr.shape[0]*image_arr.shape[1]*image_arr.shape[2]
        differences = count_differences_util(image_arr, previous_frame_global)
        if differences / float(total_values) < SKIP_OCR_THRESH:
            return # current frame very similar to previous. No need to rerun OCR

    Image.fromarray(image_arr).save(IMG_FILENAME)
    command = [OCR_BIN,
               IMG_FILENAME,
               "-x", OUTPUT_FILENAME,
              ]

    pipe = sp.Popen(command, stdout = sp.PIPE)
    pipe.wait()

    previous_frame_global = image_arr


# x is the left border (x-coordinate) of the char bounding box in the source image (in pixels).
# y is the top border (y-coordinate).
# w is the width of the bounding box.
# h is the height of the bounding box.
# Should be called after image_ocr
def ocr_output_to_coords():
    with open(OUTPUT_FILENAME) as output_file:
        # we only want the first line
        for line in output_file:
            if line[:6] == "line 1":
                break

        coords = []
        chords = []
        ocr_results = ""
        for line in output_file:
            line = line.strip()
            if len(line) >=5 and line[:6] == "line 2": # subsequent lines are lyrics
                break
            if line.find("' ") >=0: # the character read is a space. Skip the line
                continue
            ocr_char = QUOTE_REGEX.search(line)
            if ocr_char:
                ocr_results += ocr_char.group(1)
                box = BOX_REGEX.match(line)
                coords.append([int(x) for x in box.groups()])

        # combine boxes based on chords
        chord_boxes = []
        for note_match in NOTE_REGEX.finditer(ocr_results):
            chords.append(note_match.group(0))
            chord_boxes.append(combine_boxes(*coords[note_match.start():note_match.end()]))

        return chords, chord_boxes


# calculates the ratio of black pixels to non white pixels within each box of the image
def black_non_white_ratio(boxes, image_arr):
    ratios = []
    for box in boxes:
        sub_section = trunc_image(image_arr, *box)
        black_count, non_white_count =  black_count_non_white_count_util(sub_section)
        ratios.append((black_count + 1.)/(non_white_count + 1.)) # add 1 to avoid division by 0
    return ratios

# combines boxes into the smallest box containing all of them. See ocr_output_to_coords for box format
def combine_boxes(*boxes):
    if len(boxes) == 1:
        return boxes[0]
    else:
        box1, box2 = boxes[0], boxes[1]

        X = [box1[0], box2[0]]
        W = [box1[2], box2[2]]
        Y = [box1[1], box2[1]]
        H = [box1[3], box2[3]]

        W = max(X[0] + W[0], X[1] + W[1]) - min(X)
        H = max(Y[0] + H[0], Y[1] + H[1]) - min(Y)

        return combine_boxes([min(X),
                              min(Y),
                              W,
                              H],
                             *boxes[2:])

################################################
######### Cleaning Active Chords List ##########
################################################

# Convert list of active chord for each frame, to a list of intervals. Format:
# ((start_frame_nb, end_frame_nb), active_chord)
def frame_chords_to_ervals(frame_chords):
    ervals = OrderedDict()
    previous_chord = None
    chord_start = 0
    for i, c in enumerate(frame_chords):
        if c != previous_chord and previous_chord:
            ervals[(chord_start, i-1)] = previous_chord
            previous_chord = c
            chord_start = i
        if not previous_chord and c:
            previous_chord = c
            chord_start = i
    return ervals

# Remove chord intervals that are shorter than MIN_CHORD_LENGTH
# Modifying is done in place
def filter_ervals(chord_intervals, video_fps):
    # filter small intervals
    to_remove = []
    for chord_start, chord_end in ervals.iterkeys():
        if (chord_end - chord_start) < MIN_CHORD_LENGTH * video_fps:
            to_remove.append((chord_start, chord_end))
    for k in to_remove:
        ervals.pop(k)

# Merge chord intervals that are only separated by False values, for a duration less than
# MIN_CHORD_MERGE_LENGTH
def merge_chord_intervals(chord_intervals, video_fps):
    merged_chord_intervals = OrderedDict()
    chord_intervals_iter = chord_intervals.iteritems()
    (previous_chord_start, previous_chord_end), previous_chord = next(chord_intervals_iter)
    merged_chord_intervals[(previous_chord_start, previous_chord_end)] = previous_chord
    for (chord_start, chord_end), chord in chord_intervals_iter:
        if previous_chord == chord and (chord_start - previous_chord_end) < MIN_CHORD_MERGE_LENGTH * video_fps:
            merged_chord_intervals.pop((previous_chord_start, previous_chord_end))
            merged_chord_intervals[(previous_chord_start, chord_end)] = chord
            previous_chord_end = chord_end
        else:
            merged_chord_intervals[(chord_start, chord_end)] = chord
            previous_chord_start, previous_chord_end = chord_start, chord_end
            previous_chord = chord
    return merged_chord_intervals

################################################
######## Audio Extraction and Splitting ########
################################################

# use ffmpeg to create wav file from video
def extract_audio_from_video(video_file):
    audio_file = get_file_name(video_file) + ".wav"
    command = ["ffmpeg",
               "-i", video_file,
               "-ac", "1", #make the output mono
               "-vn", audio_file, #only audio
              ]
    pipe = sp.Popen(command, stdout = sp.PIPE)
    pipe.wait()
    return audio_file

# converts a video frame index into an audio sample index
def video_to_audio_frame(video_frame, video_nb_frames, audio_nb_frames):
    return int(round(video_frame / video_nb_frames * audio_nb_frames))

# returns audio frames corresponding to the given video frames
def read_audio(audio_file, video_start, video_end, video_last, video_nb_frames, audio_nb_frames):
    audio_file.readframes(video_to_audio_frame(video_start - video_last, video_nb_frames, audio_nb_frames))
    return audio_file.readframes(video_to_audio_frame(video_end - video_start, video_nb_frames, audio_nb_frames))

def get_audio_params(audio_file):
    return (audio_file.getnchannels(), audio_file.getsampwidth(),
            audio_file.getframerate(), audio_file.getnframes(),
            audio_file.getcomptype(), audio_file.getcompname())

def get_file_name(f):
    return os.path.splitext(os.path.basename(f))[0]

# Writes the audio samples to a wav file with name:
# {CHORD}_{NUMBER}.wav
# audioparams = (nchannels, sampwidth, framerate, nframes, comptype, compname)
def write_audio(audio_samples, output_dir, chord, audio_params, chord_counter = None):
    name = output_dir + chord
    if chord_counter:
        name += "_" + str(chord_counter[chord])
    name += ".wav"
    f = wave.open(name, 'w')
    f.setparams(audio_params)
    f.writeframes(audio_samples)
    f.close()

# Creates a wav file for each chord interval
def chord_intervals_to_audio_snips(chord_intervals, wav_file, output_dir, video_nb_frames):
    audio = wave.open(wav_file)
    last_frame = 0
    chord_counter = Counter()
    audio_params = get_audio_params(audio)
    audio_nb_frames = float(audio_params[3])

    # read frames for delay
    audio.readframes(int(DELAY * audio_params[2]))

    for (chord_start, chord_end), chord in chord_intervals.iteritems():
        chord_counter[chord] += 1
        chord_audio = read_audio(audio, chord_start, chord_end, last_frame, video_nb_frames, audio_nb_frames)
        last_frame = chord_end
        write_audio(chord_audio, output_dir, chord, audio_params, chord_counter)

################################################
# Functions for faster performance using numba #
################################################

# returns the count of black pixels (0) and non white pixels (!= 255), after calculating the mean of the 3rd dimension
def black_count_non_white_count(arr):
    black_count = 0
    non_white_count = 0
    D0, D1, D2 = arr.shape
    for i in range(D0):
        for j in range(D1):
            val = 0
            for k in range(D2):
                val += arr[i,j,k]
            val /= 3
            if val != 255:
                non_white_count += 1
                if val == 0:
                    black_count += 1
    return non_white_count, black_count

def count_differences(arr1, arr2):
    count = 0
    D0, D1, D2 = arr1.shape
    for i in range(D0):
        for j in range(D1):
            for k in range(D2):
                if arr1[i,j,k] != arr2[i,j,k]:
                    count += 1
    return count

# this function will appear in the profiler
def black_count_non_white_count_util(arr):
    return black_count_non_white_count_numba(arr)

# this function will appear in the profiler
def count_differences_util(arr1, arr2):
    return count_differences_numba(arr1, arr2)

black_count_non_white_count_numba = jit(black_count_non_white_count)
count_differences_numba = jit(count_differences)

################################################
############## Utility functions ###############
################################################

# delete all files created and reset previous_frame global variable.
def cleanup():
    global previous_frame_global
    previous_frame_global = None
    os.remove(IMG_FILENAME)
    os.remove(OUTPUT_FILENAME)
    wav_files = [f for f in os.listdir('.') if '.wav' in f]
    for f in wav_files:
        os.remove(f)

# Reads n frames from the OpenCV VideoCapture object.
def skip_frames(vidcap, n):
    count = 0
    while count < n:
        vidcap.read()
        count += 1

def log(string, verbose):
    if verbose:
        print string

def video_to_audio_snips(video_file, output_dir, verbose = False):
    log("Reading " + video_file, verbose)
    vidcap = cv2.VideoCapture(video_file)
    video_fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    video_nb_frames = vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    if vidcap.isOpened():
        log("Reading frame chords...", verbose)
        frame_chords = vidcap_to_frame_chords(vidcap, video_fps)
        log("Finished reading frame chords.", verbose)
        log("Creating chords intervals...", verbose)
        chord_intervals = frame_chords_to_chord_intervals(frame_chords)
        chord_intervals = merge_chord_intervals(chord_intervals, video_fps)
        filter_chord_intervals(chord_intervals, video_fps)
        log("Finished creating chord intervals.", verbose)
        log("Extracting audio from video...", verbose)
        wav_file = extract_audio_from_video(video_file)
        log("Finished extracting audio from video.", verbose)
        log("Creating audio snips...", verbose)
        chord_intervals_to_audio_snips(chord_intervals, wav_file, output_dir, video_nb_frames)
        log("Finished creating audio snips.", verbose)
        log("Cleaning up...", verbose)
        cleanup()
    return

def video_list_to_audio_snips(video_list, verbose = False):
    try:
        os.mkdir(OUTPUT_DIR)
    except OSError: #directory already exists
        pass

    for video in video_list:
        log("Now reading " + video, verbose)
        output_dir = OUTPUT_DIR + os.sep + get_file_name(video) + '/'
        log("Output directory " + output_dir, verbose)
        try:
            os.mkdir(output_dir)
        except OSError: #directory already exists, clear contents
            log("Output directory not empty. Deleting files", verbose)
            files = os.listdir(output_dir)#json.dumps(output_dir))
            for f in files:
                os.remove(output_dir + f)
            pass
        video_to_audio_snips(video, output_dir, verbose)

