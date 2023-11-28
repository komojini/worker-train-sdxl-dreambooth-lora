import os
import base64
import zipfile
import shutil
import requests
import subprocess
import torch
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry

import runpod
from runpod.serverless.utils import rp_download, upload_file_to_bucket, upload_in_memory_object
from runpod.serverless.utils.rp_validator import validate

from train_dreambooth_lora_sdxl import (
    parse_args as parse_train_args,
    main as train_dreambooth_lora,
)
from config import *
from rp_schemas import TRAIN_SCHEMA, S3_SCHEMA


torch.cuda.empty_cache()


def download_and_preprocess_images(images_url: str, output_dir: Path | str):

    downloaded_images: dict = rp_download.file(images_url)

    output_dir = str(output_dir)

    for root, dirs, files in os.walk(downloaded_images["extracted_path"]):
        if "__MACOSX" in root:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in [".jpg", ".jpeg", ".png"]:
                shutil.copy(
                    os.path.join(downloaded_images["extracted_path"], file_path),
                    output_dir,
                )
def train(
        instance_data_dir: str, 
        output_dir: str,
        class_name: str, 
        instance_token: str,
        max_train_step: int = 1000, 
        checkpointing_steps: int = 200,
        learning_rate: float = 1e-4,
        mixed_precision: str = "fp16",
        gradient_accumulation_steps: int = 1,
        lr_scheduler: str = "constant",
        resume_from_checkpoint: str = "latest",
        rank: int = 8,
        checkpoints_total_limit: int = 3,
    ):
 
    instance_prompt = f"A photo of {instance_token} {class_name},"

    train_input_args = [
        f"--pretrained_model_name_or_path={MODEL_PATH}",
        f"--instance_data_dir={instance_data_dir}",
        f"--output_dir={output_dir}",
        f"--mixed_precision={mixed_precision}",
        f"--instance_prompt='{instance_prompt}'",
        f"--class_prompt='{class_name}'",
        "--center_crop",
        f"--resume_from_checkpoint={resume_from_checkpoint}",
        f"--resolution={RESOLUTION}",
        f"--train_batch_size={TRAIN_BATCH_SIZE}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--learning_rate={learning_rate}",
        f"--lr_scheduler={lr_scheduler}",
        f"--lr_warmup_steps={LR_WARMUP_STEPS}",
        f"--checkpointing_steps={checkpointing_steps}",
        f"--max_train_steps={max_train_step}",
        f"--seed={SEED}",
        f"--rank={rank}",
        f"--checkpoints_total_limit={checkpoints_total_limit}",
        "--enable_xformers_memory_efficient_attention",
    ]

    print("Input args:", train_input_args)

    train_args = parse_train_args(
        input_args=train_input_args
    )
    train_dreambooth_lora(train_args)



def handler(job):
    
    job_input = job['input']
    lora_model_id = job_input.get('lora_model_id', None)
    if lora_model_id is None:
        lora_model_id = base64.b64encode(os.urandom(9)).decode('utf-8')
    job_input["lora_model_id"] = lora_model_id

    # -------------------------------- Validation -------------------------------- #
    # Validate the training input
    if 'train' not in job_input:
        return {"error": "Missing training input."}
    
    validated_train_input = validate(job_input['train'], TRAIN_SCHEMA)
    if 'errors' in validated_train_input:
        return {"error": validated_train_input['errors']}
    train_input = validated_train_input['validated_input']


    # Validate the S3 config, if provided
    s3_config = None
    if 's3Config' in job:
        validated_s3_config = validate(job['s3Config'], S3_SCHEMA)
        if 'errors' in validated_s3_config:
            return {"error": validated_s3_config['errors']}
        s3_config = validated_s3_config['validated_input']
    
    job_output = {}

    tmp_dir = Path("/tmp")
    output_dir = tmp_dir / lora_model_id / "output"
    instance_data_dir = tmp_dir / lora_model_id / "instance_data"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(instance_data_dir):
        shutil.rmtree(instance_data_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(instance_data_dir, exist_ok=True)

    # Download and preprocess images
    download_and_preprocess_images(train_input["data_url"], instance_data_dir)

    # Train the model
    train(
        instance_data_dir=instance_data_dir,
        output_dir=output_dir,
        class_name=train_input['class_name'],
        instance_token=train_input['instance_token'],
        max_train_step=train_input['max_train_step'],
        checkpointing_steps=train_input['checkpointing_steps'],
        learning_rate=train_input['learning_rate'],
        mixed_precision=train_input['mixed_precision'],
        gradient_accumulation_steps=train_input['gradient_accumulation_steps'],
        lr_scheduler=train_input['lr_scheduler'],
        resume_from_checkpoint=train_input['resume_from_checkpoint'],
        rank=train_input['rank'],
        checkpoints_total_limit=train_input['checkpoints_total_limit'],
    )

    # --------------------------------- Upload ----------------------------------- #
    # Zip the model
    # Get the mos recent checkpoint
    dirs = os.listdir(output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    latest_checkpoint_name = dirs[-1] if len(dirs) > 0 else None
    latest_checkpoint_path = output_dir / latest_checkpoint_name

    if latest_checkpoint_path is not None:
        archive_path = shutil.make_archive(
            base_name=lora_model_id,
            format="zip",
            root_dir=output_dir,
            base_dir=latest_checkpoint_path,
        )
    if "s3Config" in job:
        # Upload the model to S3
        job_output["train"] = {}
        job_output["train"]["checkpoint_url"] = upload_file_to_bucket(
            file_name=f"{lora_model_id}.zip",
            file_location=archive_path,
            bucket_creds=s3_config,
            bucket_name=job["s3Config"]["bucketName"],
        )
    
    return job_output


runpod.serverless.start({"handler": handler, "refresh_worker": True})

