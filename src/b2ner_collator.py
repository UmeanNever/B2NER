import logging
import os
import torch
from transformers.data.data_collator import *


logger = logging.getLogger(__name__)

SUPPORTED_DECODER_MODELS = ['codegen', 'bloo', 'gpt-neox', 'baichuan', 'internlm', 'moss', 'llama-3']
SUPPORTED_SEQ2SEQ_MODELS = ['t5', 'flan-t5']
MAX_SAMPLE_CNTS = 100


def check_model(model_name, supported_models):
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True

    return False


@dataclass
class DataCollatorForB2NER:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_dataset_name: bool = False
    common_dataset_name: str = None
    text_only: bool = False
    input_record_file: str = None
    saved_samples_cnt: int = 0

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        model_name = self.model.config._name_or_path
        # print(model_name)
        if check_model(model_name, SUPPORTED_DECODER_MODELS):
            model_inputs = self.decoder_call(batch, return_tensors)
        elif check_model(model_name, SUPPORTED_SEQ2SEQ_MODELS):
            model_inputs = self.seq2seq_call(batch, return_tensors)
        else:
            # print("Unknown model {}! Treat as decoder model.".format(model_name))
            model_inputs = self.decoder_call(batch, return_tensors)
            # raise ValueError('Unsupport model {}!'.format(model_name))

        return model_inputs

    def get_instruction(self, instance):
        # "instructions \n options \n {0} \n Answer: "
        instruction = instance['Instance']["instruction"]
        content = instance['Instance']['sentence']

        # add task/ds prefix
        prefix = ''
        if self.add_task_name:
            prefix += "Task:" + instance['Task'] + '\n'
        if self.add_dataset_name:
            ds_name = self.common_dataset_name if self.common_dataset_name else instance['Dataset']
            prefix = prefix + "Dataset:"
            prefix = prefix + ds_name + '\n' if prefix else instance['Dataset'] + '\n'
        if prefix:
            instruction = prefix + instruction

        # TODO, fix bug
        try:
            instruction = instruction.format(content)
        finally:
            return instruction


    def seq2seq_call(self, batch, return_tensors):
        sources = []
        labels = []

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            source = instruction
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))

        # TODO, support online demo
        if self.text_only:
            model_inputs = {"inputs": sources, "labels": labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    labels,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    return_tensors=return_tensors,
                    truncation=True,
                    pad_to_multiple_of=self.pad_to_multiple_of
                )
            label_mask = labels["attention_mask"].bool()
            model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

            # prepare decoder_input_ids
            if self.model is not None:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
                model_inputs["decoder_input_ids"] = decoder_input_ids

            self._save_samples(model_inputs, sources, labels)

        return model_inputs

    def decoder_call(self, batch, return_tensors):
        self.tokenizer.padding_side = 'left'
        sources = []
        label_lens = []
        labels = []
        max_len = -1
        if batch[0]['subset'] == "train":
            # limit_input_len = self.max_source_length + self.max_target_length # we padd with max_source_length below so this should be wrong
            limit_input_len = self.max_source_length
        else:
            limit_input_len = self.max_source_length

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            # add bos and eos
            raw_task_input = self.tokenizer.bos_token + instruction
            label = label + self.tokenizer.eos_token

            exemplar_str = ""
            task_input = raw_task_input.replace("@@examples@@",exemplar_str)
            num_few_shot_exemplars = len(instance["Samples"])
            
            # add few-shot exemplars
            cnt_fs_exemplars = 0
            while cnt_fs_exemplars<num_few_shot_exemplars:
                exemplar_str = exemplar_str + instance["Samples"][cnt_fs_exemplars]["instruction"] + instance["Samples"][cnt_fs_exemplars]["label"] + " \n\n"
                tmp_task_input = raw_task_input.replace("@@examples@@",exemplar_str)
                tokenized_input = self.tokenizer(tmp_task_input)["input_ids"]
                tokenized_label = self.tokenizer(label)["input_ids"]
                if instance['subset'] in ['dev', 'test'] and len(tokenized_input) <= limit_input_len:
                    task_input = tmp_task_input
                    cnt_fs_exemplars += 1
                elif len(tokenized_input) + len(tokenized_label) <= limit_input_len:
                    task_input = tmp_task_input
                    cnt_fs_exemplars += 1
                else:
                    break
                    
            tokenized_input = self.tokenizer(task_input)["input_ids"]
            tokenized_label = self.tokenizer(label)["input_ids"]
            # print(f"***********************************输入数据********************************************")
            # print("input:", raw_task_input)
            # print("label:", label)
            # print("tokenized_input:", tokenized_input)
            # print("tokenized_label:", tokenized_label)

            # (input) for inference, (input + label) for training
            if instance['subset'] in ['test']:
                label_lens.append(0)
                if len(tokenized_input) <= limit_input_len:
                    max_len = max(len(tokenized_input), max_len)
                    sources.append(task_input)
                else:
                    max_len = limit_input_len
                    # when inference, truncate from the end
                    input_wo_label = self.tokenizer.decode(
                        tokenized_input[-limit_input_len:],
                        skip_special_tokens=False
                    )
                    sources.append(input_wo_label)
                    # mark the instance id as -1
                    instance['Instance']['id'] = "-1"
            else:
                if len(tokenized_input) + len(tokenized_label) <= limit_input_len:
                    max_len = max(len(tokenized_input) + len(tokenized_label), max_len)
                    label_lens.append(len(tokenized_label))
                    sources.append(task_input + label)
                else:
                    max_len = self.max_source_length
                    input_w_label = self.tokenizer.decode(
                        (tokenized_input + tokenized_label)[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_w_label)
                    label_lens.append(max(0, limit_input_len - len(tokenized_input)))

        # TODO, support online demo
        if self.text_only:
            model_inputs = {"inputs": sources, 'labels': labels}
        else:
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )


            label_mask = model_inputs["attention_mask"].bool()
            model_inputs["labels"] = model_inputs['input_ids'].masked_fill(~label_mask, self.label_pad_token_id)
            
            max_len = min(max_len, limit_input_len)
            for k, label_len in enumerate(label_lens):
                model_inputs["labels"][k, : max_len - label_len - 1] = self.label_pad_token_id

            # loss mask
            # max_len = min(max_len, limit_input_len)
            # loss_mask = torch.ones((label_mask.shape))
            # for k, label_len in enumerate(label_lens):
            #     loss_mask[k, : max_len - label_len - 1] = 0
            # model_inputs['loss_mask'] = loss_mask.masked_fill(~label_mask, 0)

            self._save_samples(model_inputs, sources, labels)
        return model_inputs

    def _save_samples(self, model_inputs, sources, labels):
        if not self.input_record_file or self.saved_samples_cnt >= MAX_SAMPLE_CNTS:
            return

        # loss_label = []
        # if hasattr(model_inputs, 'loss_mask'):
        #     for loss, id in zip(model_inputs.loss_mask, model_inputs.input_ids):
        #         loss_label.append(self.tokenizer.decode((loss * id).view(-1).int()))

        os.makedirs(os.path.dirname(self.input_record_file), exist_ok=True)
        with open(self.input_record_file, 'a+', encoding='utf-8') as f:
            for text, label, mask_label in zip(sources, labels, model_inputs["labels"]):
                f.write(text+'\n')
                f.write(label + '\n')
                f.write(str(mask_label.int()) + '\n\n')
                self.saved_samples_cnt += 1
        # else:
        #     with open(self.input_record_file, 'a+', encoding='utf-8') as f:
        #         for text, label in zip(sources, labels['input_ids']):
        #             f.write(text + '\n')
        #             f.write(self.tokenizer.decode(label, clean_up_tokenization_spaces=False) + '\n')
        #             self.saved_samples_cnt += 1
                    