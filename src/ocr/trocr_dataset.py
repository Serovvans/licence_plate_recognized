import torch
from torchvision import transforms

AUGMENTATION = transforms.Compose([
    transforms.RandomRotation(degrees=8, fill=255),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=255),
])


def make_preprocess_fn(processor, max_length: int, augment: bool = True):
    def preprocess(batch):
        images = [
            AUGMENTATION(img.convert("RGB")) if augment else img.convert("RGB")
            for img in batch["image"]
        ]
        inputs = processor(images=images, return_tensors="pt")

        labels = processor.tokenizer(
            batch["text"],
            padding="max_length",
            max_length=max_length,
        ).input_ids

        labels = [
            [(l if l != processor.tokenizer.pad_token_id else -100) for l in label]
            for label in labels
        ]
        inputs["labels"] = torch.tensor(labels)
        return inputs

    return preprocess