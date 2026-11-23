import torch
from TSAnnotate.Load import load_labels, load_model
from torchvision.transforms.v2.functional import to_image
from torchvision.transforms.v2 import Compose, Pad, Resize, ToDtype


def annotate(image, labels_path=None, model_path=None):
    labels = load_labels(labels_path)
    model = load_model(model_path)
    image_tensor = to_image(image)
    # pad image to square and scale to target size
    width = image_tensor.size(-1)
    height = image_tensor.size(-2)
    longer_edge = max(width, height)
    pad_width = int(longer_edge - width)
    pad_height = int(longer_edge - height)
    pad_transform = Compose([Pad([pad_width, pad_height, 0, 0]), Resize([640, 640]), ToDtype(torch.float)])
    padded_image = torch.unsqueeze(pad_transform(image_tensor), 0) / 255
    # apply model
    prediction = model(padded_image, verbose=False)
    # convert box coordinates for original image
    boxes = prediction[0].boxes.data[:, 0:4] * (longer_edge / 640)
    boxes[:, 0] = torch.round(((boxes[:, 0] - pad_width) / width) * 100, decimals=2)
    boxes[:, 1] = torch.round(((boxes[:, 1] - pad_height) / height) * 100, decimals=2)
    boxes[:, 2] = torch.round(((boxes[:, 2] - pad_width) / width) * 100, decimals=2)
    boxes[:, 3] = torch.round(((boxes[:, 3] - pad_height) / height) * 100, decimals=2)
    # save annotations
    results = []
    for i in range(int(boxes.size(0))):
        label_name = labels[int(prediction[0].boxes.cls[i].item())][1]
        box_width = boxes[i, 2] - boxes[i, 0]
        box_height = boxes[i, 3] - boxes[i, 1]
        results.append({"type": "rectanglelabels",
                        "value": {"rotation": 0, "x": boxes[i, 0].item(), "y": boxes[i, 1].item(),
                                  "width": box_width.item(), "height": box_height.item(),
                                  "rectanglelabels": [label_name], },
                        "from_name": "label", "to_name": "image"})
    return results
