import os
import json
import argparse
from TSAnnotate.Annotate import annotate
from TSAnnotate.Sample import sample_video
from TSAnnotate.Filter import filter_similar
from TSAnnotate.Export import export, export_yolo_meta, export_labeling_interface
from TSAnnotate.Load import load_paths, load_image, is_video, is_image, sort_paths


parser = argparse.ArgumentParser(prog="TSAnnotate",
                                 description="Quickly annotate images for traffic sign detection and recognition by "
                                             "using pre-trained models.")
parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
parser.add_argument("task", type=str, choices=["label", "export"], help="Task to run")
parser.add_argument("input_path", help="Path to input, can be directory or file")
parser.add_argument("-o", "--output_path", default="output", help="Output directory or file")
parser.add_argument("-sr", "--sample_rate", default=1, help="Rate at which to extract images from videos")
parser.add_argument("-d", "--data_path", default=None,
                    help="Where to look for images and annotations when export task is run")
parser.add_argument("-vs", "--val_split", default=0.1, type=float,
                    help="Fraction of images to use for validation")
parser.add_argument("-ts", "--test_split", default=0.1, type=float,
                    help="Fraction of images to use for testing")
parser.add_argument("-br", "--background_ratio", default=0.1, type=float,
                    help="Maximum ratio of background images in each split")
parser.add_argument("-n", "--name", default="new_dataset", dest="dataset_name",
                    help="Name of the exported dataset")
parser.add_argument("-l", "--labels", default=None, dest="labels_path", help="Path to custom labels file")
parser.add_argument("-m", "--model", default=None, dest="model_path", help="Path to custom model")
parser.add_argument("-fs", "--filter_similar", default=None, const=0.8, type=float, nargs="?",
                    help="Filter out successive images that exceed the given similarity threshold")
parser.add_argument("-si", "--sort_inputs", action="store_true",
                    help="Sort inputs by numbers extracted from file names; numbers must be separated by '_' or '-'")


def main():
    args = parser.parse_args()
    # ensure output dirs exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.task == "label":
        if not os.path.exists(os.path.join(args.output_path, "images")):
            os.makedirs(os.path.join(args.output_path, "images"))
        if not os.path.exists(os.path.join(args.output_path, "annotations")):
            os.makedirs(os.path.join(args.output_path, "annotations"))
        input_paths = load_paths(args.input_path)
        # sort paths in flag is set
        if args.sort_inputs:
            input_paths = sort_paths(input_paths)
        print("Collected " + str(len(input_paths)) + " files")
        num_images = 0
        i = 1
        prev_image = None
        filtered_images = 0
        for path in input_paths:
            print(f"\rProcessing file {i}/{len(input_paths)}", end="", flush=True)
            if is_image(path):
                image = load_image(path)
                # filter out similar images if argument is set
                if args.filter_similar is not None:
                    if i == 1 and not args.sort_inputs:
                        print("Warning: input files should be sorted for reliable filtering")
                    if prev_image is not None:
                        if filter_similar(prev_image, image, threshold=args.filter_similar):
                            filtered_images += 1
                            continue
                prev_image = image
                annotations = annotate(image, args.labels_path, args.model_path)
                # save image
                image.save(os.path.join(args.output_path, "images", str(num_images) + ".jpg"))
                # save annotations
                task = {"id": str(num_images), "data": {"image": "/data/local-files/?d=images/" + str(num_images) + ".jpg"},
                        "annotations": [{"result": annotations}]}
                json.dump(task, open(os.path.join(args.output_path, "annotations", str(num_images) + ".json"), "w"))
                num_images += 1
            elif is_video(path):
                for image in sample_video(path, sample_rate=float(args.sample_rate)):
                    # filter out similar images if argument is set
                    if args.filter_similar is not None and prev_image is not None:
                        if filter_similar(prev_image, image, threshold=args.filter_similar):
                            filtered_images += 1
                            continue
                    prev_image = image
                    annotations = annotate(image, args.labels_path, args.model_path)
                    # save image
                    image.save(os.path.join(args.output_path, "images", str(num_images) + ".jpg"))
                    # save annotations
                    task = {"id": str(num_images),
                            "data": {"image": "/data/local-files/?d=images/" + str(num_images) + ".jpg"},
                            "annotations": [{"result": annotations}]}
                    json.dump(task, open(os.path.join(args.output_path, "annotations", str(num_images) + ".json"), "w"))
                    num_images += 1
            i += 1
        print()
        if args.filter_similar is not None:
            print("Filtered out " + str(filtered_images) + " images due to similarity")
        export_labeling_interface(args.output_path, args.labels_path)
    elif args.task == "export":
        # check where images and annotations are stored
        data_path = None
        # if data_path is set, check there first
        if args.data_path is not None:
            if not os.path.exists(os.path.join(args.data_path, "images")):
                raise RuntimeError("No images found in data path: " + args.data_path)
            if not os.path.exists(os.path.join(args.data_path, "annotations")):
                raise RuntimeError("No annotations found in data path: " + args.data_path)
            data_path = args.data_path
        # otherwise, check in output path
        elif os.path.exists(os.path.join(args.output_path, "images")) and os.path.exists(os.path.join(args.output_path, "annotations")):
            data_path = args.output_path
        # if unsuccessful, check default output path
        elif args.output_path != "output" and os.path.exists(os.path.join("output", "images")) and os.path.exists(os.path.join("output", "annotations")):
            data_path = "output"
        # if still not found, check Label Studio local files root; only works if it was launched from same shell
        elif os.environ.get("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT") is not None:
            label_studio_root = str(os.environ.get("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"))
            if os.path.exists(os.path.join(label_studio_root, "images")) and os.path.exists(os.path.join(label_studio_root, "annotations")):
                data_path = label_studio_root
        # check if data path was found
        if data_path is None:
            raise RuntimeError("Image and annotation data could not be found. Use the 'data_path' argument to define "
                               "the output directory generated by the 'label' task containing the 'images' and "
                               "'annotations' directories. If 'data_path' was set, check if it points to the correct "
                               "directory.")
        export(args.input_path, data_path, args.output_path, args.val_split, args.test_split, args.background_ratio, args.dataset_name, args.labels_path)
        export_yolo_meta(args.output_path, args.dataset_name, args.labels_path, args.val_split, args.test_split)
    return 0