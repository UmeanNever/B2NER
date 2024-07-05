#!/usr/bin/env python
# coding=utf-8

"""
Fine-tuning the library models with custom datasets
"""

import logging
import os
import sys

import datasets
import numpy as np
from datasets import load_dataset

import torch
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed, )
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, PeftModel

from b2ner_collator import DataCollatorForB2NER
from b2ner_dataset import gen_cache_path
from b2ner_trainer import B2NERTrainer, DenserEvalCallback, get_compute_metrics_fn
from utils import (handle_token_position_label_smooth, save_args, 
                   get_model_class_and_alter_tokenizer)
from arguments import ModelArguments, DataTrainingArguments, B2NERTrainingArguments, LoraArguments

# off wandb
# os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, B2NERTrainingArguments, LoraArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, lora_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)
    
    # Save arguments
    save_args(training_args, data_args, model_args, lora_args)
    
    new_tokens = None  # Currently not used.

    # Get the B2NER dataset
    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "b2ner_dataset.py"),
        data_dir=data_args.data_dir,
        task_config_dir=data_args.task_config_dir,
        instruction_file=data_args.instruction_file,
        instruction_strategy=data_args.instruction_strategy,
        cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        specific_dataset=data_args.specific_dataset,
        add_dataset_name=data_args.add_dataset_name,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        max_num_instances_per_test_task=data_args.max_num_instances_per_test_task,
        num_examples=data_args.num_examples,
        num_examples_test=data_args.num_examples_test,
        train_0shot_prop=data_args.train_0shot_prop,
        train_fewshot_prop=data_args.train_fewshot_prop,
        over_sampling=data_args.over_sampling,
        down_sampling=data_args.down_sampling,
        lang=data_args.lang,
        label_description = data_args.label_description,
        dynamic_range = data_args.dynamic_range,
        droplabel_rate = data_args.droplabel_rate,
        label_shuffle = data_args.label_shuffle,
        description_paraphrase = data_args.description_paraphrase,
        candidates_sampling = data_args.candidates_sampling,
        max_label_in_prompt = data_args.max_label_in_prompt,
    )
    raw_datasets.cleanup_cache_files()

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_class = AutoConfig
    config = config_class.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
        add_bos_token=False,
        padding_side='left'
    )

    model_class = get_model_class_and_alter_tokenizer(model_args, tokenizer)
    assert tokenizer.padding_side == "left", "The tokenizer should have `padding_side` set to `left`"
        
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True
    )
    
    handle_token_position_label_smooth(model_args, data_args, training_args, new_tokens, tokenizer, model, logger)

    # LoRA
    if training_args.use_lora or lora_args.lora_weight_path:
        if lora_args.lora_weight_path:
            logger.info("*** Loading LoRA model from %s ***", lora_args.lora_weight_path)
            logger.info("Remember to check torch_dtype for lora and original model!!!")

            # set is_trainable to False to avoid updating the loaded LoRA model
            lora_is_trainable = False if training_args.do_train else True
            model = PeftModel.from_pretrained(
                model,
                lora_args.lora_weight_path,
                is_trainable=lora_is_trainable,
                torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32 # debugged this for 2 days lol
            )
            
        if training_args.do_train:
            # if model is loaded from lora_weight_path, we need to merge and unload it before applying new LoRA
            if isinstance(model, PeftModel):
                model.merge_and_unload()
            
            if training_args.use_lora:
                lora_config = LoraConfig(
                    r=lora_args.lora_r,
                    lora_alpha=lora_args.lora_alpha,
                    target_modules=lora_args.lora_target_modules,
                    lora_dropout=lora_args.lora_dropout,
                    bias=lora_args.lora_bias,
                    task_type="CAUSAL_LM",
                )

                if lora_args.lora_target_modules == ["unk"]:
                    if 'baichuan' in model_args.model_name_or_path.lower():
                        lora_config.target_modules = ["W_pack"]
                    elif 'internlm' in model_args.model_name_or_path.lower():
                        lora_config.target_modules = ["wqkv"]
                        # lora_config.target_modules = ["wqkv", "wo", "w1", "w2", "w3"]
                    else:
                        lora_config.target_modules = ["q_proj", "k_proj", "v_proj"]
                model = get_peft_model(model, lora_config)
                
                if training_args.gradient_checkpointing:
                    model.enable_input_require_grads()
            else:
                for param in model.parameters():
                    param.requires_grad = True
        
        if training_args.deepspeed is not None and training_args.local_rank == 0:
            model.print_trainable_parameters()

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        if data_args.predict_with_train:
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))
            predict_dataset = train_dataset

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForB2NER(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        common_dataset_name=data_args.common_dataset_name,
        input_record_file=data_args.input_record_file
    )
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        if "some_new_models" in model_args.model_name_or_path.lower():
            model.gradient_checkpointing = True
        else:
            model.gradient_checkpointing_enable()

    trainer = B2NERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )

    all_metrics = {"run_name": training_args.run_name}
    # for decoder-only model, generation_max_length actually means max_new_tokens
    max_new_tokens = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    repetition_penalty = data_args.repetition_penalty

    # Training
    # num_train_epochs from training_args
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        if training_args.predict_with_generate:
            trainer.compute_metrics = get_compute_metrics_fn(model, data_args, training_args, tokenizer)
            # generate for predict dataset for each evaluation phase if predict_each_epoch
            if training_args.predict_each_epoch:
                trainer.eval_dataset = predict_dataset 
            trainer._gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": tokenizer.pad_token_id
            }
        logger.info(f"*** Run name: {training_args.run_name} ***")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # attribute 'step' Saves the tokenizer too for easy upload
        
        # if training_args.use_lora:
        #     model.save_pretrained(training_args.output_dir + "/lora")

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)

    # Evaluation
    results = {}

    if training_args.do_train and training_args.predict_with_generate:
        logger.info("*** Skip Prediction As It Was Done During Training ***")
    elif training_args.do_predict:
        logger.info(f"*** Prediction: {training_args.run_name}***")
        logger.info("*** Loading CheckPoint ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # without resume ckpt, wou  ld predict with current model
        if checkpoint:
            model = model_class.from_pretrained(checkpoint, trust_remote_code=True)
            trainer = B2NERTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=get_compute_metrics_fn(model, data_args, training_args, tokenizer),
                callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
            )

        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        trainer.compute_metrics = get_compute_metrics_fn(model, data_args, training_args, tokenizer)
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )
        
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)


    if training_args.do_demo:
        logger.info("Serving the model as a demo...")
        user_input = ''
        while True:
            user_input = input("Please enter your input to the model, or enter 'quit' to exit: ")
            if user_input.lower() == "quit":
                break
            inputs = tokenizer([user_input], return_tensors="pt")
            _, preds, _ = trainer.prediction_step(model, inputs=inputs, prediction_loss_only=False)
            print(f"Model generates: {tokenizer.decode(preds[0], skip_special_tokens=True)}\n\n")

    return results


if __name__ == "__main__":
    main()