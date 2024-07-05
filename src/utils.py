import os
import json
import torch
import dataclasses
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    )

from model.bloom import BloomForCausalLM_WithLoss
from model.codegen import CodeGenForCausalLM_WithLoss
from model.gpt_neox import GPTNeoXForCausalLM_WithLoss

def save_args(training_args, data_args, model_args, lora_args):
    # merge args into one dict and save to output dir
    if training_args.output_dir:
        os.makedirs(training_args.output_dir, exist_ok=True)
        all_args = {}
        all_args["training_args"] = dataclasses.asdict(training_args)
        all_args["data_args"] = dataclasses.asdict(data_args)
        all_args["model_args"] = dataclasses.asdict(model_args)
        all_args["lora_args"] = dataclasses.asdict(lora_args)
        with open(os.path.join(training_args.output_dir, "args.json"), "w") as f:
            json.dump(all_args, f, indent=4)


def handle_token_position_label_smooth(model_args, data_args, training_args, new_tokens, tokenizer, model, logger):
    # currently not used
    if training_args.do_train and new_tokens is not None:
        logger.info(f"Resizing tokenizer embedding layer from {len(tokenizer)} to {model.config.vocab_size}")
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        if 'baichuan' in model_args.model_name_or_path.lower():
            old_wte = model.get_input_embeddings().weight
            old_lm_head_w = model.get_output_embeddings().weight
            old_vocab_size = model.config.vocab_size
            new_vocab_size = len(tokenizer)
            new_vocab_size = ((new_vocab_size + 7) // 8) * 8  # pad to multiple of 8
            new_wte = torch.zeros(new_vocab_size, old_wte.shape[1], device=old_wte.device, dtype=old_wte.dtype)
            new_wte[:old_vocab_size, :] = old_wte
            new_lm_head_w = torch.zeros(new_vocab_size, old_lm_head_w.shape[1], device=old_lm_head_w.device, dtype=old_lm_head_w.dtype)
            new_lm_head_w[:old_vocab_size, :] = old_lm_head_w
            model.get_input_embeddings().weight = torch.nn.Parameter(new_wte)
            model.get_output_embeddings().weight = torch.nn.Parameter(new_lm_head_w)
            model.config.vocab_size = new_vocab_size
            model.vocab_size = new_vocab_size
            # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None)
        else:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None)

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )


def get_model_class_and_alter_tokenizer(model_args, tokenizer):
    if 'bloo' in model_args.model_name_or_path.lower():
        model_class = BloomForCausalLM_WithLoss
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    elif 'codegen' in model_args.model_name_or_path.lower():
        model_class = CodeGenForCausalLM_WithLoss
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    elif 'neox' in model_args.model_name_or_path.lower():  # add neox
        model_class = GPTNeoXForCausalLM_WithLoss
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    elif 't5' in model_args.model_name_or_path.lower():
        model_class = AutoModelForSeq2SeqLM
    elif 'baichuan' in model_args.model_name_or_path.lower():
        model_class = AutoModelForCausalLM
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    elif "qwen"  in model_args.model_name_or_path.lower():
        model_class = AutoModelForCausalLM
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.bos_token = ''
        tokenizer.padding_side = 'left'
    else:
        model_class = AutoModelForCausalLM
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    return model_class