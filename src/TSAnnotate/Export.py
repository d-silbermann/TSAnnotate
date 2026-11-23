import os
import json
from PIL import Image
from TSAnnotate.Load import load_labels


def export(input_path, data_path, output_path="output", val_split=0.1, test_split=0.1, background_ratio=0.1, dataset_name="new_dataset", labels_path=None):
    output_path = os.path.join(output_path, dataset_name)
    annotations = json.load(open(input_path, "r"))
    # map labels names to class ids
    short_labels = {}
    i = 0
    for _, short_label, _ in load_labels(labels_path):
        short_labels[short_label] = i
        i += 1
    # ensure output dirs exist
    if not os.path.isdir(os.path.join(output_path, "images", "train")):
        os.makedirs(os.path.join(output_path, "images", "train"))
    if val_split > 0 and not os.path.isdir(os.path.join(output_path, "images", "val")):
        os.makedirs(os.path.join(output_path, "images", "val"))
    if test_split > 0 and not os.path.isdir(os.path.join(output_path, "images", "test")):
        os.makedirs(os.path.join(output_path, "images", "test"))
    if not os.path.isdir(os.path.join(output_path, "labels", "train")):
        os.makedirs(os.path.join(output_path, "labels", "train"))
    if val_split > 0 and not os.path.isdir(os.path.join(output_path, "labels", "val")):
        os.makedirs(os.path.join(output_path, "labels", "val"))
    if test_split > 0 and not os.path.isdir(os.path.join(output_path, "labels", "test")):
        os.makedirs(os.path.join(output_path, "labels", "test"))
    if not os.path.isdir(os.path.join(output_path, "excluded")):
        os.makedirs(os.path.join(output_path, "excluded"))
    # keep track of images in each split
    train_images = 0
    train_background = 0
    val_images = 0
    val_background = 0
    test_images = 0
    test_background = 0
    excluded = 0
    # iterate over all images
    i = 1
    for task in annotations:
        print(f"\rProcessing image {i}/{len(annotations)}", end="", flush=True)
        i += 1
        image_name = task["data"]["image"].split("/")[-1]
        image = Image.open(os.path.join(data_path, "images", image_name))
        signs = []
        # iterate over annotated signs
        for result in task["annotations"][0]["result"]:
            image_width = image.width
            image_height = image.height
            # get abs pixel values for bounding box
            left = int((result["value"]["x"] / 100) * image_width)
            top = int((result["value"]["y"] / 100) * image_height)
            right = left + ((result["value"]["width"] / 100) * image_width)
            bottom = top + ((result["value"]["height"] / 100) * image_height)
            # convert into YOLO coordinates
            center_x = int((left + right) / 2) / image_width
            center_y = int((top + bottom) / 2) / image_height
            width = (right - left) / image_width
            height = (bottom - top) / image_height
            # map short label name to class number
            label = short_labels[result["value"]["rectanglelabels"][0]]
            signs.append(str(label) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height))
        # determine which split to assign image to
        if val_images + 1 <= (train_images + val_images + test_images + 1) * val_split:
            split = "val"
        elif test_images + 1 <= (train_images + val_images + test_images + 1) * test_split:
            split = "test"
        else:
            split = "train"
        # check if image is background and whether to include it
        image_output_path = os.path.join(output_path, "images", split, image_name)
        annotation_output_path = os.path.join(output_path, "labels", split, image_name.split(".")[0] + ".txt")
        if len(signs) == 0:
            # include background in train split if ratio not exceeded
            if split == "train" and train_background + 1 <= train_images * background_ratio:
                image.save(image_output_path)
                with open(annotation_output_path, "w") as f:
                    for sign in signs:
                        f.write(sign + "\n")
                train_images += 1
                train_background += 1
            # include background in val split if ratio not exceeded
            elif split == "val" and val_background + 1 <= val_images * background_ratio:
                image.save(image_output_path)
                with open(annotation_output_path, "w") as f:
                    for sign in signs:
                        f.write(sign + "\n")
                val_images += 1
                val_background += 1
            # include background in test split if ratio not exceeded
            elif split == "test" and test_background + 1 <= test_images * background_ratio:
                image.save(image_output_path)
                with open(annotation_output_path, "w") as f:
                    for sign in signs:
                        f.write(sign + "\n")
                test_images += 1
                test_background += 1
            # otherwise exclude background
            else:
                image.save(os.path.join(output_path, "excluded", image_name))
                excluded += 1
        # always include images with signs
        else:
            image.save(image_output_path)
            with open(annotation_output_path, "w") as f:
                for sign in signs:
                    f.write(sign + "\n")
            if split == "train":
                train_images += 1
            elif split == "val":
                val_images += 1
            elif split == "test":
                test_images += 1
    print()
    print("Exported " + str(train_images + val_images + test_images) + " images, excluded " + str(excluded))
    print("Train split has " + str(train_images) + " images (" + str(train_background) + " background)")
    print("Val split has " + str(val_images) + " images (" + str(val_background) + " background)")
    print("Test split has " + str(test_images) + " images (" + str(test_background) + " background)")


def export_labeling_interface(output_path, labels_path=None):
    labels = load_labels(labels_path)
    with open(os.path.join(output_path, "labeling_interface.html"), "w") as f:
        f.write("<View>\n")
        f.write("<Image name=\"image\" value=\"$image\"/>\n")
        f.write("<RectangleLabels name=\"label\" toName=\"image\">\n")
        for _, short_label, color in labels:
            f.write("<Label value=\"" + short_label + "\" background=\"" + color + "\"/>\n")
        f.write("</RectangleLabels>\n")
        f.write("</View>\n")


def export_yolo_meta(output_path="output", dataset_name="new_dataset", labels_path=None, val_split=0.1, test_split=0.1):
    with open(os.path.join(output_path, dataset_name, dataset_name + ".yaml"), "w", encoding="utf-8") as f:
        f.write("path: " + dataset_name + "\n")
        f.write("train: images/train" + "\n")
        if val_split > 0:
            f.write("val: images/val" + "\n")
        if test_split > 0:
            f.write("test: images/test" + "\n\n")
        f.write("names:" + "\n")
        i = 0
        for label, _, __ in load_labels(labels_path):
            f.write("  " + str(i) + ": " + label + "\n")
            i += 1