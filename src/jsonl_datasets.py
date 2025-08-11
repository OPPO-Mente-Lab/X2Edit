from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import random
import torch

Image.MAX_IMAGE_PIXELS = None

def multiple_16(num: float):
    return int(round(num / 16) * 16)

def get_random_resolution(min_size=512, max_size=1280, multiple=16):
    resolution = random.randint(min_size // multiple, max_size // multiple) * multiple
    return resolution

def load_image_safely(image_path, size):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print("file error: "+image_path)
        with open("failed_images.txt", "a") as f:
            f.write(f"{image_path}\n")
        return Image.new("RGB", (size, size), (255, 255, 255))
    
def make_train_dataset(args, tokenizer, accelerator=None):
    if args.train_data_dir is not None:
        print("load_data")
        dataset = load_dataset('json', data_files=args.train_data_dir)

    column_names = dataset["train"].column_names
    
    # 6. Get the column names for input/target.
    caption_column = args.caption_column
    target_column = args.target_column
    if args.subject_column is not None:
        subject_columns = args.subject_column.split(",")
    if args.spatial_column is not None:
        spatial_columns= args.spatial_column.split(",")
    
    size = args.cond_size
    noise_size = get_random_resolution(max_size=args.noise_size) # maybe 768 or higher
    subject_cond_train_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.resize((
                multiple_16(size * img.size[0] / max(img.size)),
                multiple_16(size * img.size[1] / max(img.size))
            ), resample=Image.BILINEAR)),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomRotation(degrees=20),
            transforms.Lambda(lambda img: transforms.Pad(
                padding=(
                    int((size - img.size[0]) / 2),
                    int((size - img.size[1]) / 2),
                    int((size - img.size[0]) / 2),
                    int((size - img.size[1]) / 2) 
                ),
                fill=0 
            )(img)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    cond_train_transforms = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def train_transforms(image, noise_size):
        train_transforms_ = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.resize((
                    multiple_16(noise_size * img.size[0] / max(img.size)),
                    multiple_16(noise_size * img.size[1] / max(img.size))
                ), resample=Image.BILINEAR)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        transformed_image = train_transforms_(image)
        return transformed_image
    
    def load_and_transform_cond_images(images):
        transformed_images = [cond_train_transforms(image) for image in images]
        concatenated_image = torch.cat(transformed_images, dim=1)
        return concatenated_image
    
    def load_and_transform_subject_images(images):
        transformed_images = [subject_cond_train_transforms(image) for image in images]
        concatenated_image = torch.cat(transformed_images, dim=1)
        return concatenated_image
    
    tokenizer_clip = tokenizer[0]
    tokenizer_t5 = tokenizer[1]

    def tokenize_prompt_clip_t5(examples):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                if random.random() < 0.1:
                    captions.append(" ")  # 将文本设为空
                else:
                    captions.append(caption)
            elif isinstance(caption, list):
                # take a random caption if there are multiple
                if random.random() < 0.1:
                    captions.append(" ")
                else:
                    captions.append(random.choice(caption))
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        text_inputs = tokenizer_clip(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_1 = text_inputs.input_ids

        text_inputs = tokenizer_t5(
            captions,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs.input_ids
        return text_input_ids_1, text_input_ids_2

    def preprocess_train(examples):
        _examples = {}
        if args.subject_column is not None:
            subject_images = [[load_image_safely(examples[column][i], args.cond_size) for column in subject_columns] for i in range(len(examples[target_column]))]
            _examples["subject_pixel_values"] = [load_and_transform_subject_images(subject) for subject in subject_images]
        if args.spatial_column is not None:
            spatial_images = [[load_image_safely(examples[column][i], args.cond_size) for column in spatial_columns] for i in range(len(examples[target_column]))]
            _examples["cond_pixel_values"] = [load_and_transform_cond_images(spatial) for spatial in spatial_images]
        target_images = [load_image_safely(image_path, args.cond_size) for image_path in examples[target_column]]
        _examples["pixel_values"] = [train_transforms(image, noise_size) for image in target_images]
        _examples["token_ids_clip"], _examples["token_ids_t5"] = tokenize_prompt_clip_t5(examples)
        return _examples

    if accelerator is not None:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(preprocess_train)
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    if examples[0].get("cond_pixel_values") is not None:
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        cond_pixel_values = None
    if examples[0].get("subject_pixel_values") is not None: 
        subject_pixel_values = torch.stack([example["subject_pixel_values"] for example in examples])
        subject_pixel_values = subject_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        subject_pixel_values = None

    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    token_ids_clip = torch.stack([torch.tensor(example["token_ids_clip"]) for example in examples])
    token_ids_t5 = torch.stack([torch.tensor(example["token_ids_t5"]) for example in examples])

    return {
        "cond_pixel_values": cond_pixel_values,
        "subject_pixel_values": subject_pixel_values,
        "pixel_values": target_pixel_values,
        "text_ids_1": token_ids_clip,
        "text_ids_2": token_ids_t5,
    }