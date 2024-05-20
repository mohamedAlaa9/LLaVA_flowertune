import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft.utils import prepare_model_for_kbit_training
import sys
sys.path.append('/content/LLaVA_flowertune/LLaVA')
print(sys.path)

from llava.model.language_model.llava_llama import *
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
)
import math


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_cfg.name,
    #     quantization_config=quantization_config,
    #     torch_dtype=torch.bfloat16,
    # )
    model = LlavaLlamaForCausalLM.from_pretrained(model_cfg.name,
                                                      quantization_config=quantization_config,
                                                      torch_dtype=torch.float16)

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )
    # model_name = get_model_name_from_path(model_cfg.name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_cfg.name, None, model_name
    # )

    # peft_config = LoraConfig(
    #     r=model_cfg.lora.peft_lora_r,
    #     lora_alpha=model_cfg.lora.peft_lora_alpha,
    #     lora_dropout=0.075,
    #     task_type="CAUSAL_LM",
    # )
    # import re
    # pattern = r'\((\w+)\): Linear'
    # linear_layers = re.findall(pattern, str(model.modules))
    # target_modules = list(set(linear_layers))
    # peft_config = LoraConfig(
    #     r=64,
    #     lora_alpha=16,
    #     target_modules=target_modules
    # )
    peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.075,
            # bias=model_cfg["lora_bias"],
            task_type="CAUSAL_LM",
        )

    return get_peft_model(model, peft_config)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)