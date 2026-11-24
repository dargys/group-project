from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from PIL import Image, ImageOps

from src.cnn_model import FruitCNNDropout
from src.preprocess import load_image


# Class names must match the order used during training (alphabetical).
# Note: the dataset folder is spelled "honneydew_melon"; the app uses the correct
# spelling "honeydew_melon". The order is the same, so predictions are unaffected.
CLASS_NAMES = [
    "apple",
    "honeydew_melon",
    "kiwi",
    "nectarine",
    "onion",
    "plum",
    "potato",
    "spanish_pear",
    "watermelon"
]


# Locate the saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "best_cnn_dropout_model.pth"


def load_cnn_model():
    # Create the same CNN architecture used during training
    model = FruitCNNDropout(num_classes=len(CLASS_NAMES))

    # Load the saved model weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    # Set the model to evaluation mode
    model.eval()

    return model


def predict_image(image, model):
    # Phone photos are often stored sideways together with an orientation tag.
    # Turn the photo upright using that tag, then convert it to RGB.
    image = ImageOps.exif_transpose(image).convert("RGB")

    # Save a temporary image for the shared preprocessing function
    with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        image.save(temp_path)

        # Apply the same preprocessing used during training
        processed_image = load_image(temp_path)

    finally:
        temp_path.unlink(missing_ok=True)

    # Convert the image to a PyTorch tensor
    image_tensor = torch.from_numpy(
        np.asarray(processed_image, dtype=np.float32)
    ).permute(2, 0, 1).unsqueeze(0)

    # Predict without updating the model
    with torch.no_grad():
        outputs = model(image_tensor)

        # Convert model outputs into probabilities
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Get the predicted class
    predicted_index = probabilities.argmax().item()
    predicted_class = CLASS_NAMES[predicted_index]

    # Get probabilities for all classes
    class_probabilities = {
        class_name: round(probability.item() * 100, 2)
        for class_name, probability in zip(
            CLASS_NAMES, probabilities
        )
    }

    return predicted_class, class_probabilities