
TRAIN_SCHEMA = {
    "lora_model_id": {
        "type": str,
        "required": False,
        "default": None
    },
    "data_url": {
        "type": str,
        "required": True
    },
    "class_name": {
        "type": str,
        "required": True
    },
    "instance_token": {
        "type": str,
        "required": True
    },
    "max_train_step": {
        "type": int,
        "required": False,
        "default": 1000
    },
    "checkpointing_steps": {
        "type": int,
        "required": False,
        "default": 200
    },
    "learning_rate": {
        "type": float,
        "required": False,
        "default": 1e-4
    },
    "mixed_precision": {
        "type": str,
        "required": False,
        "default": "fp16",
        "constraints": lambda x: x in ["fp16", "fp32"],
    },
    "gradient_accumulation_steps": {
        "type": int,
        "required": False,
        "default": 1
    },
    "lr_scheduler": {
        "type": str,
        "required": False,
        "default": "constant",
        'constraints': lambda scheduler: scheduler in ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']
    },
    "resume_from_checkpoint": {
        "type": str,
        "required": False,
        "default": "latest",
        'constraints': lambda checkpoint: checkpoint in ['latest', 'best']
    },
    "rank": {
        "type": int,
        "required": False,
        "default": 8
    },
    "checkpoints_total_limit": {
        "type": int,
        "required": False,
        "default": 3
    }
}

S3_SCHEMA = {
    'accessId': {
        'type': str,
        'required': True
    },
    'accessSecret': {
        'type': str,
        'required': True
    },
    'bucketName': {
        'type': str,
        'required': True
    },
    'endpointUrl': {
        'type': str,
        'required': True
    }
}