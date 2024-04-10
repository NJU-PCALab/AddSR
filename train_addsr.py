import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset # ''datasets'' is a library
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

import sys
sys.path.append("..")
from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from dataloaders.paired_dataset import PairedCaptionDataset

#from ADD.models.unet_discriminator import UNetDiscriminatorSN
from ADD.models.discriminator import ProjectedDiscriminator
from ADD.models.SwinIR import SwinIR
from ADD.utils.util_net import compute_hinge_loss, EMA
from ADD.models.vit import vit_large, vit_small
import ADD.utils.util_net as util_net

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

if is_wandb_available():
    import wandb

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def TA_weight(t):
    if t == 999:
        weight = 0.85
    elif t == 749:
        weight = 1
    elif t == 499:
        weight = 1.2
    else:
        weight = 1.4
    return weight

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)

from torchvision import transforms
tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
                ])
ram_transforms = transforms.Compose([       # 定义RAM的图像转化，对RAM的输入进行如下的变化
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:     # 加载SwinIR模型参数,移除 'module' 字符
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)

def image_grid(imgs, rows, cols):       # 将输入的多张图像，排列成行和列，组成一张图
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step):      # 生成日志
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):  # 根据模型的名字，import对应的模型
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="preset/models/stable-diffusion-2-base",
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path_Tea",
        type=str,
        default='preset/seesr',
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_model_name_or_path_Tea",
        type=str,
        default='preset/seesr',
        help="Path to pretrained unet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path_Stu",
        type=str,
        default='preset/seesr',
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_model_name_or_path_Stu",
        type=str,
        default='preset/seesr',
        help="Path to pretrained unet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(    # 模型输出结果的保存路径
        "--output_dir",
        type=str,
        default="/root/workspace/ruixie/DataSet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=50000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(        # 每更新多少步保存一次网络参数，以及结果
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(    # 是否从某个checkpoint继续进行训练
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='/home/notebook/data/group/LSDIR/HR_sub',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=50000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="SeeSR",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument("--root_folders",  type=str, default='')
    parser.add_argument("--null_text_ratio", type=float, default=0.5)
    parser.add_argument("--use_ram_encoder", action='store_true')
    parser.add_argument("--ram_ft_path", type=str, default='preset/models/DAPE.pth')
    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["image_attentions"])

    
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

def tokenize_captions_null(gt, tokenizer, is_train=True):
        captions = []
        for caption in range(gt.shape[0]):
            captions.append("")
            
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

def make_train_dataset(args, tokenizer, accelerator):   # 将数据读入，生成训练数据集
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )


    def tokenize_captions(examples, is_train=True):  # 将输入的文本转化为token
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images   # 高分辨率图像
        examples["conditioning_pixel_values"] = conditioning_images  # 低分辨率图像
        examples["input_ids"] = tokenize_captions(examples)     # text_embed

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    gt = torch.stack([example["gt"] for example in examples])
    gt = gt.to(memory_format=torch.contiguous_format).float()

    kernel1 = torch.stack([example["kernel1"] for example in examples])
    kernel1 = kernel1.to(memory_format=torch.contiguous_format).float()

    kernel2 = torch.stack([example["kernel2"] for example in examples])
    kernel2 = kernel2.to(memory_format=torch.contiguous_format).float()

    sinc_kernel = torch.stack([example["sinc_kernel"] for example in examples])
    sinc_kernel = sinc_kernel.to(memory_format=torch.contiguous_format).float()

    gt_path = [example["gt_path"] for example in examples]

    return {
        "gt": gt,
        "kernel1": kernel1,
        "kernel2": kernel2,
        "sinc_kernel": sinc_kernel,
        "gt_path": gt_path,
    }

# def main(args):
args = parse_args()
logging_dir = Path(args.output_dir, args.logging_dir)


from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    kwargs_handlers=[ddp_kwargs]
)

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Handle the repository creation
if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

# Load the tokenizer
if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
elif args.pretrained_model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

# import correct text encoder class
text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = text_encoder_cls.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
# unet = UNet2DConditionModel.from_pretrained(
#     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
# )

# 实例化教师模型的unet
if not args.unet_model_name_or_path_Tea:
    # resume from SD
    logger.info("Loading unet weights from SD")
    unet = UNet2DConditionModel.from_pretrained_orig(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, use_image_cross_attention=args.use_ram_encoder
    )
else:
    # resume from self-train
    logger.info("Loading unet weights from self-train")
    unet = UNet2DConditionModel.from_pretrained(
        args.unet_model_name_or_path_Tea, subfolder="unet", revision=args.revision
    )
    print(f'===== if use ram encoder? {unet.config.use_image_cross_attention}')

# 实例化教师模型的controlnet
if args.controlnet_model_name_or_path_Tea:
    # resume from self-train
    logger.info("Loading existing controlnet weights")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path_Tea, subfolder="controlnet", low_cpu_mem_usage=False)

else:
    logger.info("Initializing controlnet weights from unet")
    controlnet = ControlNetModel.from_unet(unet, use_image_cross_attention=True)

# 实例化学生模型的unet
if not args.unet_model_name_or_path_Stu:
    # resume from SD
    logger.info("Loading unet weights from SD")
    unet_stu = UNet2DConditionModel.from_pretrained_orig(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, use_image_cross_attention=args.use_ram_encoder
    )
else:
    # resume from self-train
    logger.info("Loading unet weights from self-train")
    unet_stu = UNet2DConditionModel.from_pretrained(
        args.unet_model_name_or_path_Stu, subfolder="unet", revision=args.revision
    )
    print(f'===== if use ram encoder? {unet_stu.config.use_image_cross_attention}')

# 实例化学生模型的controlnet
if args.controlnet_model_name_or_path_Stu:
    # resume from self-train
    logger.info("Loading existing controlnet weights")
    controlnet_stu = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path_Stu, subfolder="controlnet", low_cpu_mem_usage=False)

else:
    logger.info("Initializing controlnet weights from unet")
    controlnet_stu = ControlNetModel.from_unet(unet_stu, use_image_cross_attention=True)
    

# `accelerate` 0.16.0 will have better support for customized saving
if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        #i = len(weights) - 1

        # while len(weights) > 0:
        #     weights.pop()
        #     model = models[i]

        #     sub_dir = "controlnet"
        #     model.save_pretrained(os.path.join(output_dir, sub_dir))

        #     i -= 1
        #assert len(models) == 2 and len(weights) == 2
        i = 0
        for model in models:
            if isinstance(model, UNet2DConditionModel) or isinstance(model, ControlNetModel):
                sub_dir = "unet_" + str(i) if isinstance(model, UNet2DConditionModel) else "controlnet_" + str(i)
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
                i += 1

    def load_model_hook(models, input_dir):

        assert len(models) == 2
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if not isinstance(model, UNet2DConditionModel):
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True

            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)
controlnet.requires_grad_(False)

unet_stu.requires_grad_(False)
controlnet_stu.requires_grad_(False)

ema_unet = EMA(unet_stu, 0.99)
ema_control = EMA(controlnet_stu, 0.99)

## release the cross-attention part in the unet_stu.
for name, module in unet_stu.named_modules():
    if name.endswith(tuple(args.trainable_modules)):
        print(f'{name} in <unet_stu> will be optimized.')
        for params in module.parameters():
            params.requires_grad = True

for name, module in controlnet_stu.named_modules():
     if name.startswith('controlnet_cond_embedding') or name.endswith(tuple(args.trainable_modules)):
        print(f'{name} in <controlnet_stu> will be optimized.')
        for params in module.parameters():
            params.requires_grad = True

## init the RAM or DAPE model
from ram.models.ram_lora import ram
from ram import get_transform
if args.ram_ft_path is None:
    print("======== USE Original RAM ========")
else:
    print("==============")
    print(f"USE FT RAM FROM: {args.ram_ft_path}")
    print("==============")

RAM = ram(pretrained='preset/ram_swin_large_14m.pth',
            pretrained_condition=args.ram_ft_path, 
            image_size=384,
            vit='swin_l')
RAM.eval()
RAM.requires_grad_(False)

if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

        unet_stu.enable_xformers_memory_efficient_attention()
        controlnet_stu.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

if args.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()

    unet_stu.enable_gradient_checkpointing()
    controlnet_stu.enable_gradient_checkpointing()

# Check that all trainable models are in full precision
low_precision_error_string = (
    " Please make sure to always have all model weights in full float32 precision when starting training - even if"
    " doing mixed precision training, copy of the weights should still be float32."
)

if accelerator.unwrap_model(controlnet).dtype != torch.float32:
    raise ValueError(
        f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
    )
if accelerator.unwrap_model(unet).dtype != torch.float32:
    raise ValueError(
        f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
    )


# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

# Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
        )

    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

# Optimizer creation
print(f'=================Optimize ControlNet and Unet ======================')
params_to_optimize = list(unet_stu.parameters()) + list(controlnet_stu.parameters())


print(f'start to load optimizer...')


optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

# 实例化提取cls_lr的特征网络
model_fea = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1.0)
util_net.reload_model(model_fea, torch.load('weights/dinov2_vits14_pretrain.pth'))
model_fea.requires_grad_(False)

# 实例化判别器头
model_dis = ProjectedDiscriminator(c_dim=384).train()
optimizer_D = torch.optim.AdamW(
    model_dis.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

train_dataset = PairedCaptionDataset(root_folders=args.root_folders,
                                    tokenizer=tokenizer,
                                    null_text_ratio=args.null_text_ratio,
                                    )

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    num_workers=args.dataloader_num_workers,
    batch_size=args.train_batch_size,
    #prefetch_factor=4,
    shuffle=False
)
    

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    num_training_steps=args.max_train_steps * accelerator.num_processes,
    num_cycles=args.lr_num_cycles,
    power=args.lr_power,
)

# Prepare everything with our `accelerator`.
controlnet, unet, optimizer, train_dataloader, lr_scheduler, controlnet_stu, unet_stu, model_fea, model_dis, optimizer_D = accelerator.prepare(
    controlnet, unet, optimizer, train_dataloader, lr_scheduler,
    controlnet_stu, unet_stu, model_fea, model_dis, optimizer_D,
)

# For mixed precision training we cast the text_encoder and vae weights to half-precision
# as these models are only used for inference, keeping weights in full precision is not required.
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# Move vae, unet and text_encoder to device and cast to weight_dtype
vae.to(accelerator.device, dtype=weight_dtype)
text_encoder.to(accelerator.device, dtype=weight_dtype)
RAM.to(accelerator.device, dtype=weight_dtype)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
if accelerator.is_main_process:
    tracker_config = dict(vars(args))

    # tensorboard cannot handle list types for config
    tracker_config.pop("validation_prompt")
    tracker_config.pop("validation_image")
    tracker_config.pop("trainable_modules")

    accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

# Train!
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")
# if not isinstance(train_dataset, WebImageDataset):
#     logger.info(f"  Num examples = {len(train_dataset)}")
#     logger.info(f"  Num batches each epoch = {len(train_dataloader)}")


logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num batches each epoch = {len(train_dataloader)}")

logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")
global_step = 0
first_epoch = 0

# Potentially load in the weights and states from a previous save
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
else:
    initial_global_step = 0

progress_bar = tqdm(
    range(0, args.max_train_steps),
    initial=initial_global_step,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
)

ema_unet.register()
ema_control.register()
criterion_GAN = torch.nn.BCEWithLogitsLoss()
ram_mean = [0.485, 0.456, 0.406]
ram_std = [0.229, 0.224, 0.225]
ram_normalize = transforms.Normalize(mean=ram_mean, std=ram_std)

for epoch in range(first_epoch, args.num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        # with accelerator.accumulate(controlnet):
        with accelerator.accumulate(controlnet), accelerator.accumulate(unet), accelerator.accumulate(controlnet_stu), accelerator.accumulate(unet_stu):
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)     # HR图像
            controlnet_image = batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)  # LR图像
            
            # Convert images to latent space
            latents = vae.encode(pixel_values).latent_dist.sample()  # 将HR图像通过vae encoder嵌入到潜在空间
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            p = torch.rand((1,))
            if p > 0.5:
                target_set = torch.tensor([499, 749, 999], device=latents.device)
                random_index = torch.randint(0, len(target_set), size=(bsz,), device=latents.device)
                tt = target_set[random_index]
                t_TA = tt.long()
            else:
                target_set = torch.tensor([999], device=latents.device)
                tt = target_set.repeat(bsz)
                t_TA = torch.tensor([1000], device=latents.device).repeat(bsz).long()

            timesteps_stu = tt.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps_stu)

            # # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]     # text_embed

            # extract soft semantic label
            with torch.no_grad():
                ram_image = batch["ram_values"].to(accelerator.device, dtype=weight_dtype)
                ram_encoder_hidden_states = RAM.generate_image_embeds(ram_image)    # representation embed

                ram_image = batch["ram_values_HR"].to(accelerator.device, dtype=weight_dtype)
                ram_encoder_hidden_states_tea = RAM.generate_image_embeds(ram_image)

            ###################################
            ###---生成学生模型一步采样估计的z0---###
            ###################################

            if p > 0.5:
                with torch.no_grad():
                    down_block_res_samples_stu, mid_block_res_sample_stu, _, _ = controlnet_stu(
                        noisy_latents,
                        timesteps_stu,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                        image_encoder_hidden_states=ram_encoder_hidden_states,
                    )

                    # Predict the noise residual
                    model_pred_stu = unet_stu(
                        noisy_latents,
                        timesteps_stu,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples_stu
                        ],
                        mid_block_additional_residual=mid_block_res_sample_stu.to(dtype=weight_dtype),
                        image_encoder_hidden_states=ram_encoder_hidden_states,
                    ).sample

            else:
                down_block_res_samples_stu, mid_block_res_sample_stu, _, _ = controlnet_stu(
                    noisy_latents,
                    timesteps_stu,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                    image_encoder_hidden_states=ram_encoder_hidden_states,
                )

                # Predict the noise residual
                model_pred_stu = unet_stu(
                    noisy_latents,
                    timesteps_stu,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples_stu
                    ],
                    mid_block_additional_residual=mid_block_res_sample_stu.to(dtype=weight_dtype),
                    image_encoder_hidden_states=ram_encoder_hidden_states,
                ).sample

            z0_stu = noise_scheduler.get_z0(noisy_latents, model_pred_stu, timesteps_stu)

            # second steps
            if p > 0.5:
                indices_1 = torch.nonzero(timesteps_stu != 249).squeeze()
                z0_stu_SecStep = z0_stu[indices_1, :, :, :].detach()
                timesteps_stu_ = []
                if indices_1.shape == ():
                    indices_1 = indices_1.unsqueeze(0)
                for idx in indices_1:
                    if timesteps_stu[idx] == 999:
                        timesteps_stu_.append(torch.tensor([749], device=latents.device))
                    elif timesteps_stu[idx] == 749:
                        timesteps_stu_.append(torch.tensor([499], device=latents.device))
                    elif timesteps_stu[idx] == 499:
                        timesteps_stu_.append(torch.tensor([249], device=latents.device))
                timesteps_stu_ = torch.cat(timesteps_stu_, dim=0).squeeze(-1)

                z0_stu_SecStep_ = 1 / vae.config.scaling_factor * z0_stu_SecStep
                z0_stu_SecStep_ = z0_stu_SecStep_.to(next(vae.parameters()).dtype)
                x0_stu_SecStep = vae.decode(z0_stu_SecStep_, return_dict=False)[0].clamp(-1, 1).detach()

                ram_values = F.interpolate(x0_stu_SecStep, size=(384, 384), mode='bicubic')
                for i in range(ram_values.shape[0]):
                   ram_values[i] = ram_normalize(ram_values[i])

                # extract soft semantic label
                with torch.no_grad():
                   ram_image = ram_values.to(accelerator.device, dtype=weight_dtype)
                   ram_encoder_hidden_states_Sec = RAM.generate_image_embeds(ram_image)  # representation embed

                # (this is the forward diffusion process)
                noisy_latents_SecStep = noise_scheduler.add_noise(latents, noise, timesteps_stu_)

                down_block_res_samples_stu_, mid_block_res_sample_stu_, _, _ = controlnet_stu(
                    noisy_latents_SecStep,
                    timesteps_stu_,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=x0_stu_SecStep,
                    return_dict=False,
                    image_encoder_hidden_states=ram_encoder_hidden_states_Sec,
                )

                # Predict the noise residual
                model_pred_stu_ = unet_stu(
                    noisy_latents_SecStep,
                    timesteps_stu_,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples_stu_
                    ],
                    mid_block_additional_residual=mid_block_res_sample_stu_.to(dtype=weight_dtype),
                    image_encoder_hidden_states=ram_encoder_hidden_states_Sec,
                ).sample

                z0_stu = noise_scheduler.get_z0(noisy_latents_SecStep, model_pred_stu_, timesteps_stu_)

            ##################################
            ###---生成教师模型一步采样估计的z0---###
            ##################################

            # Sample timesteps for Teacher
            timesteps_tea = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz * 4,), device=latents.device)
            timesteps_tea = timesteps_tea.long()

            # 生成输入教师模型的中间状态
            z_t_tea_list = []
            t_tea = torch.split(timesteps_tea, dim=0, split_size_or_sections=bsz)
            for i in t_tea:
                z_t_tea_list.append(noise_scheduler.add_noise(z0_stu, noise, i))

            # 生成教师模型一步采样估计的z0
            with torch.no_grad():
                z0_tea_list = []
                counter = 0
                for i in z_t_tea_list:
                    down_block_res_samples_tea, mid_block_res_sample_tea, repre_tea, features_tea = controlnet(
                        # 将z_t, t, text_embed, LR, representation embed输入到controlnet(包括了image_encoder)中
                        i,
                        t_tea[counter],
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=pixel_values / 2 + 0.5,
                        return_dict=False,
                        image_encoder_hidden_states=ram_encoder_hidden_states_tea,
                    )

                    # Predict the noise residual
                    model_pred_tea = unet(
                        i,
                        t_tea[counter],
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples_tea
                        ],
                        mid_block_additional_residual=mid_block_res_sample_tea.to(dtype=weight_dtype),
                        image_encoder_hidden_states=ram_encoder_hidden_states_tea,
                    ).sample

                    z0_tea = noise_scheduler.get_z0(i, model_pred_tea, t_tea[counter])
                    z0_tea_list.append(z0_tea)
                    counter += 1

            # compute the discriminator loss & update parameters
            _, cls_lr = model_fea(F.interpolate(controlnet_image, size=518, mode='bilinear'))

            ### generate the x0_stu from z0_stu
            z0_stu_ = 1 / vae.config.scaling_factor * z0_stu
            z0_stu_ = z0_stu_.to(next(vae.parameters()).dtype)
            x0_stu = vae.decode(z0_stu_, return_dict=False)[0].clamp(-1, 1)

            pred_real, features = model_dis(pixel_values, cls_lr.detach())
            pred_fake, _ = model_dis(x0_stu.detach(), cls_lr.detach())
            pred_fake = torch.cat(pred_fake, dim=1)
            r1_lamda = 0

            if r1_lamda != 0:
                grad_penalty_list = []
                for i in range(0, len(pred_real)):
                    grad_real = torch.autograd.grad(outputs=pred_real[i].sum(), inputs=features[str(i)], create_graph=True)[0]
                    grad_penalty = (grad_real.contiguous().view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean() * r1_lamda
                    grad_penalty_list.append(grad_penalty)

                pred_real = torch.cat(pred_real, dim=1)
                loss_real = torch.mean(torch.relu(1.0 - pred_real)) + sum(grad_penalty_list) / len(grad_penalty_list)
            else:
                pred_real = torch.cat(pred_real, dim=1)
                loss_real = torch.mean(torch.relu(1.0 - pred_real))

            loss_fake = torch.mean(torch.relu(1.0 + pred_fake))

            loss_D = loss_real + loss_fake
            accelerator.backward(loss_D)
            optimizer_D.step()
            optimizer_D.zero_grad(set_to_none=args.set_grads_to_none)
            model_fea.zero_grad(set_to_none=args.set_grads_to_none)

            # compute the generator loss
            pred_fake, _ = model_dis(x0_stu, cls_lr.detach())
            pred_fake = torch.cat(pred_fake, dim=1)
            gan_loss = -torch.mean(pred_fake)

            # generate the weights for different t_tea
            weight = noise_scheduler.get_weights(latents, timesteps_tea)
            weight_ = torch.split(weight, split_size_or_sections=latents.size(0))

            # compute the distillation loss
            counter = 0
            dis_mse = []
            for i in z0_tea_list:
                z0_tea = 1 / vae.config.scaling_factor * i
                z0_tea = z0_tea.to(next(vae.parameters()).dtype)
                x0_tea = vae.decode(z0_tea, return_dict=False)[0].clamp(-1, 1)
                x0_tea = (x0_tea / 2 + 0.5)
                dis_mse.append((x0_tea.detach() - (x0_stu / 2 + 0.5))**2 * weight_[counter])
                counter += 1
            dis_loss = torch.stack(dis_mse, dim=-1)
            dis_loss1 = dis_loss[0].mean() * TA_weight(t_TA[0])
            dis_loss2 = dis_loss[1].mean() * TA_weight(t_TA[1])

            loss = (dis_loss1.mean() + dis_loss2.mean()) / 2 + 0.02*gan_loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = list(controlnet_stu.parameters()) + list(unet_stu.parameters())
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            ema_unet.update()
            ema_control.update()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    ema_unet.apply_shadow()
                    ema_control.apply_shadow() 
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    ema_unet.restore()
                    ema_control.restore()
                    logger.info(f"Saved state to {save_path}")

                # if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                if False:
                    image_logs = log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        controlnet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

        logs = {"D_loss": loss_D.detach().item(), "G_loss": gan_loss.detach().item(), "dis_loss": ((dis_loss1.mean() + dis_loss2.mean()) / 2).detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

# Create the pipeline using using the trained modules and save it.
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    controlnet = accelerator.unwrap_model(controlnet)
    controlnet.save_pretrained(args.output_dir)

    unet = accelerator.unwrap_model(unet)
    unet.save_pretrained(args.output_dir)

    if args.push_to_hub:
        save_model_card(
            repo_id,
            image_logs=image_logs,
            base_model=args.pretrained_model_name_or_path,
            repo_folder=args.output_dir,
        )
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )

accelerator.end_training()
