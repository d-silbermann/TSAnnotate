import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from importlib.resources import read_text
from huggingface_hub import hf_hub_download


SUPPORTED_VIDEOS = ["mov", "mp4"]
SUPPORTED_IMAGES = ["jpg", "jpeg", "png"]


def load_paths(input_path):
    if os.path.isdir(input_path):
        return sorted(load_paths_from_dir(input_path))
    else:
        if is_video(input_path) or is_image(input_path):
            return [input_path]
        else:
            raise RuntimeError("Input file type not supported: " + input_path.split(".")[-1])


def load_paths_from_dir(dir_path):
    files = []
    for path in os.listdir(dir_path):
        total_path = os.path.join(dir_path, path)
        if os.path.isdir(total_path):
            files.extend(load_paths_from_dir(total_path))
        else:
            if is_video(total_path) or is_image(total_path):
                files.append(total_path)
    return files


def load_image(image_path):
    return Image.open(image_path)


def is_video(path):
    return path.split(".")[-1] in SUPPORTED_VIDEOS


def is_image(path):
    return path.split(".")[-1] in SUPPORTED_IMAGES


def load_labels(path=None):
    labels = []
    if path is None:
        for line in read_text("TSAnnotate.data", "labels.csv", encoding="utf-8").split("\n"):
            if line.strip() != "":
                labels.append(line.strip().split(","))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() != "":
                    labels.append(line.strip().split(","))
        if len(labels) == 0:
            raise RuntimeError("No labels loaded from custom file: " + path)
    return labels


def load_model(path=None):
    if path is None:
        try:
            checkpoint = hf_hub_download(repo_id="dsilbermann/traffic_sign_detection_europe", filename="model.pt")
            return YOLO(checkpoint)
        except:
            raise RuntimeError("Failed to load default model from HuggingFace")
    elif os.path.isfile(path):
        try:
            return YOLO(path)
        except:
            raise FileNotFoundError("Model could not be loaded from " + path)
    else:
        path_split = path.split("/")
        repo_id = path_split[0] + "/" + path_split[1]
        filename = path_split[2]
        try:
            checkpoint = hf_hub_download(repo_id=repo_id, filename=filename)
            return YOLO(checkpoint)
        except:
            raise RuntimeError("Failed to load custom model from HuggingFace at " + path +
                               "; path must include filename like this: user/repo/filename.pt")


def sort_paths(paths):
    sort_by = []
    for path in paths:
        path = path.split(".")[0].split("/")[-1]
        # split file name and look for numbers to sort by
        for c1 in path.split("_"):
            found_number = False
            for c2 in c1.split("-"):
                try:
                    c_int = int(c2)
                except:
                    continue
                sort_by.append(c_int)
                found_number = True
                break
            if found_number:
                break
    sort_by = np.array(sort_by)
    sorted_indices = np.argsort(sort_by)
    result = []
    for index in sorted_indices:
        result.append(paths[index])
    return result
