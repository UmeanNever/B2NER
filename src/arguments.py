from dataclasses import dataclass, field
from typing import Optional
import typing
from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default="en", metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    specific_dataset: str = field(
        default="",
        metadata={"help": "specify the dataset in the task config file. (when finutune for specific dataset)"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation task."}
    )
    max_num_instances_per_test_task: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of instances we will consider for each test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: int = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    num_examples_test: int = field(
        default=0,
        metadata={"help": "number of in-context positive examples during test."}
    )
    max_label_in_prompt: int = field(
        default=18,
        metadata={"help": "number of maximum label in prompt for training."}
    )
    train_0shot_prop: float = field(
        default=0,
        metadata={"help": "proportion of few shot examples in the training set."}
    )
    train_fewshot_prop: float = field(
        default=0,
        metadata={"help": "proportion of few shot examples in the training set."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )
    common_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "common dataset name for zero shot."}
    )
    over_sampling: Optional[str] = field(
        default=False,
        metadata={"help": "Whether to over sampling the dataset to max_num_instances_per_task"}
    )
    down_sampling: Optional[float] = field(
        default=1.0,
        metadata={"help": "rate to down sampling the dataset after max_num_instances_per_task"}
    )
    retokenize_label: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to retokenize label with <LABEL1>..."}
    )
    label_description: Optional[str] = field(
        default=None,
        metadata={"help": "file name label description to add"}
    )
    dynamic_range: Optional[bool] = field(
        default=False,
        metadata={"help": "use dynamic range for labels in prompt in training when too many labels"}
    )
    droplabel_rate: Optional[float] = field(
        default=0.0,
        metadata={"help": "prob to drop labels in prompt in training"}
    )
    label_shuffle: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to shuffle the labels in prompt"}
    )
    description_paraphrase: Optional[int] = field(
        default=0,
        metadata={"help": "how many to use paraphrased label descriptions"}
    )
    candidates_sampling: Optional[int] = field(
        default=0,
        metadata={"help": "how many candidates to add to the label description"}
    )
    predict_with_train: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to evaluate and predict on train data"}
    )


@dataclass
class B2NERTrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    do_extract_label_emb: Optional[bool] = field(
        default=False, metadata={"help": "Whether to extract retokenized label embedding. Currently not used."}
    )
    predict_each_epoch: Optional[bool] = field(
        default=True, metadata={"help": "Whether to do predict at each epoch / evaluation step."}
    )
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use lora"}
    )


@dataclass
class LoraArguments:
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["unk"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False