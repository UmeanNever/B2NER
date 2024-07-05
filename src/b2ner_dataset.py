import json
import os
import random
import datasets
from hashlib import md5
import re

logger = datasets.logging.get_logger(__name__)
MAX_LABEL_IN_PROMPT_INFERENCE = 100
TASK_CONFIG_FILES = {"train": "train_tasks.json", "dev": "dev_tasks.json", "test": "test_tasks.json"}
INSTRUCTION_STRATEGIES = ['single', 'multiple']
ANSWER_PREFIX = {"en": "Answer:", "zh": "答案:", "de": "Antwort:", "nl": "Antwoord:", "es": "Respuesta:"}
ICL_TEXT_PREFIX = {"en": "Text: ", "zh": "文本: "}
MEANS_TEXT = {"en": "means", "zh": "是指"}
CANDIDATES_TEXT = {"zh": ",代表性的提取样例有", "en": ", representative example mentions are "}
SINGLE_QUOTES_SUBSTITUTE = "#$%#"
AUX_PROB = 0.3
ZERO_SHOT_STR = 'zero-shot'
FEW_SHOT_STR = 'few-shot'
ZERO_SHOT_WITH_DESCRIPTION_STR = 'zero-shot-with-description'

# get cache path
def gen_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_config_dir + data_args.specific_dataset + \
               data_args.instruction_file + data_args.instruction_strategy + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError('{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)

def preprocess_explanation(explanation):
    # Remove trailing "。" or "."
    explanation = re.sub(r'[。.]$', '', explanation)
    # Replace all Chinese style punctuation marks with English style
    explanation = re.sub(r'[，]', ',', explanation)
    explanation = re.sub(r'[。]', '.', explanation)
    explanation = re.sub(r'[！]', '!', explanation)
    explanation = re.sub(r'[？]', '?', explanation)
    explanation = re.sub(r'[；]', ';', explanation)
    explanation = re.sub(r'[：]', ':', explanation)
    explanation = re.sub(r'[“”]', '"', explanation)
    explanation = re.sub(r'[‘’]', "'", explanation)
    explanation = re.sub(r'[（]', '(', explanation)
    explanation = re.sub(r'[）]', ')', explanation)
    explanation = re.sub(r'[【]', '[', explanation)
    explanation = re.sub(r'[】]', ']', explanation)
    explanation = re.sub(r'[《]', '<', explanation)
    explanation = re.sub(r'[》]', '>', explanation)
    # Replace double quotes with single quotes
    explanation = re.sub(r'\"', "'", explanation)
    return explanation.strip()

class B2NERConfig(datasets.BuilderConfig):
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max dev sample size of each task
        max_num_instances_per_test_task: max test sample size of each task
    """

    def __init__(
            self,
            *args,
            data_dir=None,
            instruction_file=None,
            instruction_strategy=None,
            task_config_dir=None,
            specific_dataset=None,
            add_dataset_name=None,
            num_examples=None,
            num_examples_test=None,
            max_num_instances_per_task=None,
            max_num_instances_per_eval_task=None,
            max_num_instances_per_test_task=None,
            train_0shot_prop=None,
            train_fewshot_prop=None,
            over_sampling=None,
            down_sampling=None,
            lang=None,
            label_map=None,
            label_description=None,
            dynamic_range=None,
            droplabel_rate=None,
            label_shuffle=None,
            description_paraphrase=None,
            candidates_sampling=None,
            max_label_in_prompt=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.num_examples_test = num_examples_test
        self.specific_dataset = specific_dataset
        self.add_dataset_name = add_dataset_name
        self.train_0shot_prop = train_0shot_prop
        self.train_fewshot_prop = train_fewshot_prop
        self.over_sampling = over_sampling
        self.down_sampling = down_sampling
        self.lang = lang
        self.instructions = self._parse_instruction(instruction_file)
        self.task_configs = self._parse_task_config(task_config_dir) 
        self.instruction_strategy = instruction_strategy
        self.max_num_instances_per_task = max_num_instances_per_task
        self.max_num_instances_per_eval_task = max_num_instances_per_eval_task
        self.max_num_instances_per_test_task = max_num_instances_per_test_task
        self.label_map = label_map
        self.label_description = label_description
        self.dynamic_range = dynamic_range
        self.droplabel_rate = droplabel_rate
        self.label_shuffle = label_shuffle
        self.description_paraphrase = description_paraphrase
        self.candidates_sampling = candidates_sampling
        self.max_label_in_prompt = max_label_in_prompt
        
    # 解析instructin路径
    def _parse_instruction(self, instruction_file):
        '''
        {
            "zh": {
                "NER": [
                {"instruction_type": "zero-shot", "instruction": "给定实体的标签范围，请识别文本中属于这些标签的所有实体。答案格式为 \"实体标签: 实体; 实体标签: 实体\"。\n标签范围: {labels_str}\n\n文本: {text} \n"},
                {"instruction_type": "zero-shot-with-description", "instruction": "给定实体的标签范围以及对每个标签的描述，请识别文本中属于这些标签的所有实体。答案格式为 \"实体标签: 实体; 实体标签: 实体\"。\n标签范围: {labels_str}\n标签描述: {label_descriptions_str}\n\n文本: {text} \n"},
                {"instruction_type": "few-shot", "instruction": "给定实体的标签范围，请识别文本中属于这些标签的实体。答案格式为 \"实体标签: 实体; 实体标签: 实体\"。下面有一些例子。 \n标签范围: {labels_str} \n\n@@examples@@文本: {text} \n"}
                ]
            },
            "en": {
                "NER": [
                {"instruction_type": "zero-shot", "instruction": "Given the label set of entities, please recognize all the entities in the text. The answer format should be \"entity label: entity; entity label: entity\". \nLabel Set: {labels_str} \n\nText: {text} \n"},
                {"instruction_type": "zero-shot-with-description", "instruction": "Given the label set of entities and their descriptions, please recognize all the entities in the text. The answer format should be \"entity label: entity; entity label: entity\". \nLabel Set: {labels_str} \nLabel Description: {label_descriptions_str} \n\nText: {text} \n"},
                {"instruction_type": "few-shot", "instruction": "Given the label set of entities, please recognize all the entities in the text. The answer format should be \"entity label: entity; entity label: entity\". Here are some examples. \nLabel Set: {labels_str} \n\n@@examples@@Text: {text} \n"}
                ]
            }
        }
        '''
        if not instruction_file:
            return None

        with open(instruction_file, 'r+') as f:
            origin_ml_instructions = json.load(f)
        
        ml_instructions = {}
        for lang in origin_ml_instructions:
            origin_instructions = origin_ml_instructions[lang]
            instructions = {"zero-shot": {}, "few-shot": {}, "zero-shot-with-description": {}}
            for task in origin_instructions:
                for task_instruction in origin_instructions[task]:
                    instruct_type = task_instruction["instruction_type"]
                    if instruct_type == "zero-shot":
                        instructions['zero-shot'][task] = instructions['zero-shot'].get(task, [])
                        instructions['zero-shot'][task].append(task_instruction["instruction"])
                    elif instruct_type == "few-shot":
                        instructions['few-shot'][task] = instructions['few-shot'].get(task, [])
                        instructions['few-shot'][task].append(task_instruction["instruction"])
                    elif instruct_type == "zero-shot-with-description":
                        instructions['zero-shot-with-description'][task] = instructions['zero-shot-with-description'].get(task, [])
                        instructions['zero-shot-with-description'][task].append(task_instruction["instruction"])
                    else:
                        raise ValueError("Invalid instruction type {}, please check your instruction file {}"
                                        .format(instruct_type, instruction_file))
            ml_instructions[lang] = instructions
        
        return ml_instructions

    # 给定task_config，解析json文件，
    def _parse_task_config(self, task_config_dir):
        """
        Keys are task names, values are dictionary of dataset names and sampling strategies
        Task config file example:
            {
              "RE": [
                {"sampling strategy": "random", "dataset name": "conll04"}
              ],
              "NER": [
                {"sampling strategy": "random", "dataset name": "ACE05_coarse-grained"},
                {"sampling strategy": "full", "dataset name": "conll2003"}
              ],
              "EE": [
                {"sampling strategy": "random", "dataset name": "GENIA"}
              ]
            }
        """
        if not task_config_dir:
            return None

        task_configs = {}
        for task, file_name in TASK_CONFIG_FILES.items():
            task_config_file = os.path.join(task_config_dir, file_name)

            if not os.path.exists(task_config_file):
                raise ValueError('Please check {} config, {} not exists!'.format(task, task_config_file))

            with open(task_config_file, 'r+') as f:
                task_configs[task] = json.loads(f.read())

        return task_configs


# class for build dataset, including load data, sampling, and format prompts. Few-shot examples are also sampled here.
class B2NERInstructions(datasets.GeneratorBasedBuilder):
    """
    B2NERD Dataset.
    Main entry point: split_generators and generate_examples
    """

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = B2NERConfig
    BUILDER_CONFIGS = [
        B2NERConfig(name="default", description="Default config for B2NERD")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "Task": datasets.Value("string"),
                    "Dataset": datasets.Value("string"),
                    "Language": datasets.Value("string"),
                    "subset": datasets.Value("string"),
                    "Samples": [{
                        "id": datasets.Value("string"),
                        "instruction": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "sentence": datasets.Value("string"),
                        "label": datasets.Value("string"),
                        "instruction": datasets.Value("string"),
                        "ground_truth": datasets.Value("string")
                    }
                }
            ),
            supervised_keys=None
        )

    # The entry point for the training, validation, and test sets
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_configs is None:
            logger.error("Please provide right input: data_dir or task_config_dir!")

        # split dir save datasets
        # task config to specify train,dev,test
        split_dir = self.config.data_dir
        task_configs = self.config.task_configs

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['train'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['dev'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": split_dir,
                    "task_config": task_configs['test'],
                    "max_num_instances_per_task": self.config.max_num_instances_per_test_task,  # default load total test samples to test
                    "subset": "test"
                }),
        ]

    # load dataset json and label json
    def _load_dataset(self, dataset_path, labels_path):
        # check if file exists, set empty list if not
        try:
            check_path(dataset_path)
        except ValueError:
            logger.warning(f"Dataset file {dataset_path} not exists, please check!")
            return [], []
        with open(dataset_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)
        with open(labels_path, encoding="utf-8") as labels_f:
            labels = json.load(labels_f)

        return instances, labels
    
    def _load_label_description(self, labels_path):
        labels_ann_path = labels_path.replace('labels', self.config.label_description)
        if not os.path.exists(labels_ann_path):
            logger.warning(f"Label description file {labels_ann_path} not exists, please check!")
            return {}
        with open(labels_ann_path, encoding="utf-8") as labels_f:
            label_descriptions = json.load(labels_f)

        return label_descriptions
    
    def _load_description_paraphrase(self, labels_path):
        desc_para_path = labels_path.replace('labels', "label_description_paraphrase")
        with open(desc_para_path, encoding="utf-8") as labels_f:
            paraphrased_descriptions = json.load(labels_f)
        # preprocess paraphrased descriptions using preprocess_explanation
        for label in paraphrased_descriptions:
            paraphrased_descriptions[label] = [preprocess_explanation(desc) for desc in paraphrased_descriptions[label]]
        # only take self.config.description_paraphrase
        paraphrased_descriptions = {k: v[:self.config.description_paraphrase] for k, v in paraphrased_descriptions.items()}

        return paraphrased_descriptions
    
    def _load_entity_candidates(self, labels_path):
        entity_candidates_path = labels_path.replace('labels', 'entity_mentions')
        if not os.path.exists(entity_candidates_path):
            logger.warning(f"Entity candidates file {entity_candidates_path} not exists, please check!")
            return {}
        with open(entity_candidates_path, encoding="utf-8") as entity_f:
            entity_candidates = json.load(entity_f)
        # only take top 10
        candidate_top_k = 10
        entity_candidates = {k: list(v.keys())[:candidate_top_k] for k, v in entity_candidates.items()}

        return entity_candidates

    def _get_instruction(self, lang, task, paradigm=ZERO_SHOT_STR):
        if lang not in self.config.instructions:
            raise ValueError(f"Language {lang} not supported in the instruction file! Check your task config and instruction config!")
        assert self.config.instruction_strategy in INSTRUCTION_STRATEGIES
        task_instructions = self.config.instructions[lang][paradigm][task]
        if self.config.instruction_strategy == "single":
            return task_instructions[0]
        else:
            return random.choice(task_instructions)

    # sample dataset
    def _sampling_dataset(self, instances, sampling_strategy, max_num_instances, extra_sample_rate=1):
        if sampling_strategy.startswith('random') and max_num_instances is not None and max_num_instances >= 0:
            # Check if there is specific amount written in strategy, if so, override max_num_instances
            if sampling_strategy.startswith('random_'):
                max_num_instances = int(sampling_strategy.split('_')[1])
            # sample 'max_num_instances' instances from all instances when the number of instances is larger than 'max_num_instances'
            instances = random.sample(instances, min(len(instances), max_num_instances))
            # over sampling option to sample to max_num_instances, not used
            if max_num_instances!=None and self.config.over_sampling and len(instances) < max_num_instances:
                origin_instances = instances.copy()
                while len(instances) < max_num_instances:
                    instances.append(random.choice(origin_instances))
        
        # upsample instances, not used
        if sampling_strategy.startswith('upsample') or sampling_strategy.startswith('random&upsample'):
            instances = instances * int(sampling_strategy.split('_')[1])
            
        # downsample instances, not used
        if self.config.down_sampling < 1.0 and 'downsample' in sampling_strategy:
            instances = random.sample(instances, int(len(instances)*self.config.down_sampling))

        # extra down sample, mainly for balance zero-shot and few-shot samples
        final_max_num_instances = int(len(instances)*extra_sample_rate)
        instances = instances[:final_max_num_instances]
        return instances

    def load_NER_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset, 
                         extra_sample_rate, num_few_shot_exemplars):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        if len(labels) == 0:
            return
        
        sample_template = {"Language": lang, "Task": "NER", "Dataset": dataset_name, "Samples": [], "subset": subset}

        if self.config.label_map:
            labels = [self.config.label_map[f"{dataset_name}_{label}"] for label in labels]
        
        # load label descriptions
        if self.config.label_description:
            if 'empty' in self.config.label_description:
                label_description_dict = {label: '' for label in labels}
            else:
                label_description_dict = self._load_label_description(labels_path)
            if self.config.description_paraphrase > 0:
                paraphrased_descriptions = self._load_description_paraphrase(labels_path)
            if self.config.candidates_sampling > 0:
                entity_candidates = self._load_entity_candidates(labels_path)
            
        # sample dataset
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances, extra_sample_rate)
        
        def extract_ner_label(_instance):
            kv_pairs = []

            for entity in _instance['entities']:
                if entity['type'] == 'NA' or entity['type'] == '':
                    continue
                if self.config.label_map:
                    entity['type'] = self.config.label_map[f"{dataset_name}_{entity['type']}"]
                kv_pair = [entity['name'], entity['type']]
                kv_pairs.append(kv_pair)

            if len(kv_pairs) > 0:
                _label = " " + "; ".join(["{}: {}".format(v, k) for (k, v) in kv_pairs])
            else:
                _label = " None"
            return _label
        
        # sample few-shot exemplars
        few_shot_list = []
        if num_few_shot_exemplars > 0:
            few_shot_insts = self._sampling_dataset(instances, 'random', num_few_shot_exemplars)
            for idx, exemplar_instance in enumerate(few_shot_insts):
                instruction = ICL_TEXT_PREFIX[lang] + exemplar_instance['sentence'] + ' \n' + ANSWER_PREFIX[lang]
                label = extract_ner_label(exemplar_instance)
                few_shot_list.append({
                    "id": str(idx),
                    "instruction": instruction,
                    "label": label,
                    "ground_truth": label
                })

        def construct_sample(idx, cur_instance, cur_labels):            
            # for cross label dataset
            if 'labels' in cur_instance:
                cur_labels = cur_instance['labels']
            
            # dynamic label set when too many labels
            if self.config.dynamic_range and len(cur_labels) > self.config.max_label_in_prompt and subset == 'train':

                # count all labels in current instance
                inst_labels = [entity['type'] for entity in cur_instance['entities']]
                inst_labels = list(set(inst_labels))
                if len(inst_labels) > self.config.max_label_in_prompt:
                    inst_labels = random.sample(inst_labels, self.config.max_label_in_prompt)

                # random sample labels to fill the prompt
                sampled_cur_labels = random.sample(cur_labels, int(max(self.config.max_label_in_prompt-len(inst_labels), 0)))

                if len(inst_labels) > 0:
                    cur_labels = inst_labels + sampled_cur_labels
                    cur_labels = list(set(cur_labels))
                else:
                    cur_labels = sampled_cur_labels
            
            # random label dropout
            if self.config.droplabel_rate > 0 and subset == 'train':
                labels_dropout = [l for l in cur_labels if random.random() > self.config.droplabel_rate]
                if len(labels_dropout) == 0 and len(cur_labels) > 0:
                    labels_dropout = [random.choice(cur_labels)]
                cur_instance["entities"] = [entity for entity in cur_instance["entities"] if entity["type"] in labels_dropout]
                cur_labels = labels_dropout
            
            # label shuffle
            if self.config.label_shuffle and subset == 'train':
                labels_shuffle = random.sample(cur_labels, len(cur_labels))
                cur_labels = labels_shuffle
            
            _labels_str = ', '.join(cur_labels)
            if self.config.add_dataset_name and subset == 'train':
                _labels_str = _labels_str + " \nDataset: " + dataset_name
            if self.config.label_description and label_description_dict:
                cur_label_description_dict = label_description_dict.copy()
                if self.config.description_paraphrase > 0 and subset == 'train':
                    # random pick one from paraphrased descriptions
                    cur_label_description_dict = {label: random.choice(paraphrased_descriptions[label]) for label in cur_labels if not label[-1].isdigit()}
                if self.config.candidates_sampling > 0 and entity_candidates:
                    # random pick self.config.candidates_sampling number of candidates, currently not used
                    for label in cur_labels:
                        if label[-1].isdigit():
                            continue
                        cur_entity_candidates = entity_candidates[label]
                        if subset == 'train':
                            cur_entity_candidates = random.sample(cur_entity_candidates, min(self.config.candidates_sampling, len(cur_entity_candidates)))
                        else:
                            cur_entity_candidates = cur_entity_candidates[:self.config.candidates_sampling]
                        # wrap with single quote
                        cur_entity_candidates = [f"'{candidate}'" for candidate in cur_entity_candidates]
                        cur_label_description_dict[label] = cur_label_description_dict[label] + CANDIDATES_TEXT[lang] + ', '.join(cur_entity_candidates)
                _label_descriptions = [f"{label} {MEANS_TEXT[lang]} \"{cur_label_description_dict[label]}\"" for label in cur_labels if not label[-1].isdigit()]
                _label_descriptions_str = '; '.join(_label_descriptions)
            
            # construct the data for current instance
            example = sample_template.copy()
            example["Samples"] = few_shot_list
            if num_few_shot_exemplars > 0:
                instruction = self._get_instruction(lang, 'NER', FEW_SHOT_STR)
                instruction = instruction.format(labels_str=_labels_str, text=cur_instance['sentence']) + ANSWER_PREFIX[lang]
            elif self.config.label_description and label_description_dict:
                instruction = self._get_instruction(lang, 'NER', ZERO_SHOT_WITH_DESCRIPTION_STR)
                instruction = instruction.format(labels_str=_labels_str, label_descriptions_str=_label_descriptions_str, text=cur_instance['sentence']) + ANSWER_PREFIX[lang]
            else:
                instruction = self._get_instruction(lang, 'NER')
                instruction = instruction.format(labels_str=_labels_str, text=cur_instance['sentence']) + ANSWER_PREFIX[lang]
            
            target_text = extract_ner_label(cur_instance)
            example["Instance"] = {
                "id": str(idx),
                "sentence": cur_instance['sentence'],
                "label": target_text,
                "ground_truth": target_text,
                "instruction": instruction
            }
            return example
        
        for idx, instance in enumerate(instances):
            '''
            # Currently not used. 
            # Dynamic label for test subset, check the length of labels in current sample, if it is too long, break into several test samples
            if self.config.dynamic_range and len(labels) > MAX_LABEL_IN_PROMPT_INFERENCE and subset != 'train':
                for i in range(0, len(labels), MAX_LABEL_IN_PROMPT_INFERENCE):
                    cur_labels = labels[i:i+MAX_LABEL_IN_PROMPT_INFERENCE]
                    cur_instance = instance.copy()
                    # filter out entities with labels not in cur_labels
                    cur_instance["entities"] = [entity for entity in cur_instance["entities"] if entity["type"] in cur_labels]
                    # if no entities left, skip? Not reasonable.
                    example = construct_sample(idx, cur_instance, cur_labels)
                    yield example
            '''
            cur_labels = labels.copy()
            example = construct_sample(idx, instance, cur_labels)
            yield example
                    
    def load_ES_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                        extra_sample_rate, num_few_shot_exemplars):
        # ES = Entity Span
        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Language": lang, "Task": "ES", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction(lang, 'ES')
            instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + ANSWER_PREFIX[lang]
            entities = []

            for entity in instance['entities']:
                entities.append(entity["name"])

            if len(entities) > 0:
                label = " " + ", ".join([entity_name for entity_name in entities])
            else:
                label = " None"

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            if random.random() < AUX_PROB:
                yield example

    def load_ET_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                        extra_sample_rate, num_few_shot_exemplars):
        # ET = Entity Type
        instances, labels = self._load_dataset(dataset_path, labels_path)

        sample_template = {"Language": lang, "Task": "ET", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction(lang, 'ET')
            entities = []
            kv_pairs = []

            for entity in instance['entities']:
                if entity['type'] == 'NA' or entity['type'] == '':
                    continue
                kv_pair = [entity['name'], entity['type']]
                kv_pairs.append(kv_pair)
                entities.append(entity["name"])

            entities_str = ", ".join([entity_name for entity_name in entities])
            instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + " Entities: " + entities_str + " \n" + ANSWER_PREFIX[lang]

            if len(kv_pairs) > 0:
                label = " " + "; ".join(["{}: {}".format(v, k) for (k, v) in kv_pairs])
            else:
                label = " None"

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            if random.random() < AUX_PROB:
                yield example

    def load_EP_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                        extra_sample_rate, num_few_shot_exemplars):
        # EP = Entity Pair
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Language": lang, "Task": "EP", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction(lang, 'EP')
            instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + ANSWER_PREFIX[lang]
            relation_pairs = []
            ground_truth_pairs = []

            for relation in instance['relations']:
                if relation['type'] == 'NA' or relation['type'] == '':
                    continue
                relation_pair = [relation['head']['name'], relation['tail']['name']]
                ground_truth_pairs.append(relation_pair)
                relation_pairs.append(relation_pair)

            if len(relation_pairs) > 0:
                label = " " + "; ".join(["{}, {}".format(h, t) for (h, t) in relation_pairs])
            else:
                label = ' None'

            if len(ground_truth_pairs) > 0:
                ground_truth = " " + "; ".join(["{}, {}".format(h, t) for (h, t) in ground_truth_pairs])
            else:
                ground_truth = ' None'

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": ground_truth,
                "instruction": instruction
            }

            if random.random() < AUX_PROB:
                yield example

    def load_EPR_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                         extra_sample_rate, num_few_shot_exemplars):
        # EPR = Entity Pair Relationship
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Language": lang, "Task": "EPR", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction(lang, 'EPR')
            relation_pairs = []
            entity_pairs = []
            ground_truth_pairs = []

            for relation in instance['relations']:
                if relation['type'] == 'NA' or relation['type'] == '':
                    ground_truth_pairs.append([relation['head']['name'], 'NA', relation['tail']['name']])
                    continue
                relation_pair = [relation['head']['name'], relation['type'], relation['tail']['name']]
                entity_pair = [relation['head']['name'], relation['tail']['name']]
                ground_truth_pairs.append(relation_pair)
                relation_pairs.append(relation_pair)
                entity_pairs.append(entity_pair)

            ep_name = ' ' + "; ".join(["{}, {}".format(h, t) for (h, t) in entity_pairs])
            instruction += "Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + " Entity Pairs: " + ep_name + ' \n' + ANSWER_PREFIX[lang]

            if len(relation_pairs) > 0:
                label = ' ' + "; ".join(["{}: {}, {}".format(r, h, t) for (h, r, t) in relation_pairs])
            else:
                label = ' None'

            if len(ground_truth_pairs) > 0:
                ground_truth = ' ' + "; ".join(["{}: {}, {}".format(r, h, t) for (h, r, t) in ground_truth_pairs])
            else:
                logger.error("******Error item: {}******".format(instance))
                raise Exception('Dataset Error:{}, No ground truth!'.format(dataset_name))

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": ground_truth,
                "instruction": instruction
            }

            if random.random() < AUX_PROB:
                yield example

    def load_RE_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                        extra_sample_rate, num_few_shot_exemplars):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Language": lang, "Task": "RE", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances, extra_sample_rate)

        def extract_re_label(dataset_name, instance):
            relation_pairs = []
            ground_truth_pairs = []

            for relation in instance['relations']:
                if relation['type'] == 'NA' or relation['type'] == '':
                    # skip NA, ground_truth_pairs.append([relation['head']['name'], 'NA', relation['tail']['name']])
                    continue
                relation_pair = [relation['head']['name'], relation['type'], relation['tail']['name']]
                ground_truth_pairs.append(relation_pair)
                relation_pairs.append(relation_pair)

            if len(relation_pairs) > 0:
                label = ' ' + "; ".join("{}: {}, {}".format(r, h, t) for (h, r, t) in relation_pairs)
            else:
                label = ' None'

            if len(ground_truth_pairs) > 0:
                ground_truth = ' ' + "; ".join("{}: {}, {}".format(r, h, t) for (h, r, t) in ground_truth_pairs)
            else:
                logger.error("******Error item: {}******".format(instance))
                raise Exception('Dataset Error:{}, No ground truth!'.format(dataset_name))
            return label, ground_truth
        
        # extract fs exemplars
        few_shot_list = []
        if num_few_shot_exemplars > 0:
            few_shot_insts = self._sampling_dataset(instances, 'random', num_few_shot_exemplars)
            for idx, exemplar_instance in enumerate(few_shot_insts):
                instruction = ICL_TEXT_PREFIX[lang] + exemplar_instance['sentence'] + ' \n' + ANSWER_PREFIX[lang]
                label, ground_truth = extract_re_label(dataset_name, exemplar_instance)
                few_shot_list.append({
                    "id": str(idx),
                    "instruction": instruction,
                    "label": label,
                    "ground_truth": ground_truth
                })
        
        for idx, instance in enumerate(instances):

            cur_labels = labels.copy()
            '''
            # Currently not used
            # random label dropout
            if self.config.droplabel_rate > 0 and subset == 'train':
                labels_dropout = [l for l in cur_labels if random.random() > self.config.droplabel_rate]
                if len(labels_dropout) == 0 and len(cur_labels) > 0:
                    labels_dropout = [random.choice(cur_labels)]
                instance["entities"] = [entity for entity in instance["relations"] if entity["type"] in labels_dropout]
                cur_labels = labels_dropout
            
            # shuffle
            if self.config.label_shuffle and subset == 'train':
                labels_shuffle = random.sample(cur_labels, len(cur_labels))
                cur_labels = labels_shuffle
            '''
            labels_str = ', '.join(cur_labels)

            example = sample_template.copy()
            example["Samples"] = few_shot_list
            if num_few_shot_exemplars > 0:
                instruction = self._get_instruction(lang, 'RE', FEW_SHOT_STR)
            else:
                instruction = self._get_instruction(lang, 'RE')
            instruction = instruction.format(labels_str=labels_str, text=instance['sentence']) + ANSWER_PREFIX[lang]
            label, ground_truth = extract_re_label(dataset_name, instance)

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": ground_truth,
                "instruction": instruction
            }

            yield example

    
    def load_Sentiment_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                        extra_sample_rate, num_few_shot_exemplars):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Language": lang, "Task": "Sentiment", "Dataset": dataset_name, "Samples": [], "subset": subset}
        
        labels_str = ', '.join(labels)
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances, extra_sample_rate)

        #sentiment dataset format：{"text": , "label":}

        def extract_Sentiment_label(dataset_name, instance):
            relation_pairs = []
            ground_truth_pairs = []

            for sentiment in instance['label']:
                if sentiment == 'NA' or sentiment == '':
                    continue
                ground_truth_pairs.append(sentiment)
                relation_pairs.append(sentiment)

            label = ''.join(relation_pairs) if relation_pairs else 'None'
            ground_truth = ''.join(ground_truth_pairs) if ground_truth_pairs else 'None'
            if not ground_truth_pairs:
                logger.error("******Error item: {}******".format(instance))
                raise Exception('Dataset Error:{}, No ground truth!'.format(dataset_name))
            return label, ground_truth
        
        few_shot_list = []
        if num_few_shot_exemplars > 0:
            few_shot_insts = self._sampling_dataset(instances, 'random', num_few_shot_exemplars)
            for idx, exemplar_instance in enumerate(few_shot_insts):
                instruction = ICL_TEXT_PREFIX[lang] + exemplar_instance['label'] + ' \n' + ANSWER_PREFIX[lang]
                label, ground_truth = extract_Sentiment_label(dataset_name, exemplar_instance)
                few_shot_list.append({
                    "id": str(idx),
                    "instruction": instruction,
                    "label": label,
                    "ground_truth": ground_truth
                })
        
        for idx, instance in enumerate(instances):

            example = sample_template.copy()
            example["Samples"] = few_shot_list
            if num_few_shot_exemplars > 0:
                instruction = self._get_instruction(lang, 'Sentiment', FEW_SHOT_STR)
            else:
                instruction = self._get_instruction(lang, 'Sentiment')
            instruction = instruction.format(labels_str=labels_str, text=instance['text']) + ANSWER_PREFIX[lang]
            label, ground_truth = extract_Sentiment_label(dataset_name, instance)

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['text'],
                "label": label,
                "ground_truth": ground_truth,
                "instruction": instruction
            }

            yield example


    def load_EE_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                        extra_sample_rate, num_few_shot_exemplars):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Language": lang, "Task": "EE", "Dataset": dataset_name, "Samples": [], "subset": subset}

        # TODO, reconstruct Event Instruction to two stage
        # TODO, check
        labels_str = f'Event type: {labels[0]}, Arguments type: {labels[1]}.'
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction(lang, 'EE')
            instruction += " Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + ANSWER_PREFIX[lang]
            event_pairs = []

            if 'events' not in instance and 'event_list' in instance:
                instance['events'] = instance['event_list']
                
            for k, event in enumerate(instance['events']):
                instance['events'][k]['trigger'] = event['trigger'].replace("'", SINGLE_QUOTES_SUBSTITUTE)
                instance['events'][k]['type'] = event['type'].replace("'", SINGLE_QUOTES_SUBSTITUTE)

                if event['type'] == 'NA' or event['type'] == '':
                    continue
                event_type = event['type']
                event_trigger = event['trigger']
                
                # TODO, reverse the name and role? Not yet.
                event_arguments = [" {}: {}".format(argument['name'], argument['role']) for
                                   argument in event['arguments']]

                event_arguments = "None" if not event_arguments else ",".join(event_arguments)
                event_pair = [event_type, event_trigger, event_arguments]
                event_pairs.append(event_pair)

            if len(event_pairs) > 0:
                label = ",".join([" ( {}: {}, {}) ".format(type, trigger, arguments)
                                   for (type, trigger, arguments) in event_pairs])
            else:
                label = ' None'

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example

    def load_EET_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                         extra_sample_rate, num_few_shot_exemplars):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Language": lang, "Task": "EET", "Dataset": dataset_name, "Samples": [], "subset": subset}

        # TODO, reconstruct Event Instruction to two stage
        labels_str = ", ".join(labels.keys())
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction(lang, 'EET')
            # instruction += " Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + ANSWER_PREFIX[lang]
            instruction = instruction.format(labels_str=labels_str, text=instance['sentence']) + ANSWER_PREFIX[lang]
            event_pairs = []

            for k, event in enumerate(instance['events']):
                instance['events'][k]['trigger'] = event['trigger'].replace("'", SINGLE_QUOTES_SUBSTITUTE)
                instance['events'][k]['type'] = event['type'].replace("'", SINGLE_QUOTES_SUBSTITUTE)

                if event['type'] == 'NA' or event['type'] == '' or event['trigger'] == 'NA' or event['trigger'] == '':
                    continue
                event_type = event['type']
                event_trigger = event['trigger']
                event_pair = [event_type, event_trigger]
                # dedup
                if event_pair not in event_pairs:
                    event_pairs.append(event_pair)

            if len(event_pairs) > 0:
                label = " " + "; ".join(["{}: {}".format(type, trigger) for (type, trigger) in event_pairs])
            else:
                label = ' None'

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example

    def load_newEET_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                         extra_sample_rate, num_few_shot_exemplars):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Language": lang, "Task": "newEET", "Dataset": dataset_name, "Samples": [], "subset": subset}

        labels_str = ", ".join(labels.keys())
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            example = sample_template.copy()
            instruction = self._get_instruction(lang, 'newEET')
            # instruction += " Option: " + labels_str + " \n" + "Text: " + "{0}" + " \n" + ANSWER_PREFIX[lang]
            instruction = instruction.format(labels_str=labels_str, text=instance['sentence']) + ANSWER_PREFIX[lang]
            event_pairs = []

            for k, event in enumerate(instance['events']):
                instance['events'][k]['trigger'] = event['trigger'].replace("'", SINGLE_QUOTES_SUBSTITUTE)
                instance['events'][k]['type'] = event['type'].replace("'", SINGLE_QUOTES_SUBSTITUTE)

                if event['type'] == 'NA' or event['type'] == '' or event['trigger'] == 'NA' or event['trigger'] == '':
                    continue
                event_type = event['type']
                event_trigger = event['trigger']
                event_pair = [event_type, event_trigger]
                # dedup
                if event_pair not in event_pairs:
                    event_pairs.append(event_pair)

            if len(event_pairs) > 0:
                label = " " + "; ".join([type for (type, _) in event_pairs])
            else:
                label = ' None'

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }

            yield example

    def load_EEA_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                         extra_sample_rate, num_few_shot_exemplars):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Language": lang, "Task": "EEA", "Dataset": dataset_name, "Samples": [], "subset": subset}

        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        for idx, instance in enumerate(instances):
            if len(instance['events']) > 1:
                raise "Error: EEA dataset should only have one event."
            if len(instance['events']) == 0:
                continue # BYzhy
            labels_str = ', '.join(labels[instance['events'][0]['type']])
            example = sample_template.copy()
            instruction = self._get_instruction(lang, 'EEA')
            instruction = instruction.format(event_type=instance['events'][0]['type'], 
                                             event_trigger=instance['events'][0]['trigger'],
                                             labels_str=labels_str, text=instance['sentence']) + ANSWER_PREFIX[lang]

            event = instance['events'][0]
            
            # Reversed the name and role
            event_arguments = [" {}: {}".format(argument['role'], argument['name']) for
                               argument in event['arguments']]

            label = " None" if not event_arguments else ";".join(event_arguments)

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }
            yield example

    def load_newEEA_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset,
                         extra_sample_rate, num_few_shot_exemplars):
        instances, labels = self._load_dataset(dataset_path, labels_path)
        sample_template = {"Language": lang, "Task": "newEEA", "Dataset": dataset_name, "Samples": [], "subset": subset}

        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances)

        def extract_neweea_label(instance):
            merged_event = {'arguments': [], 'type': None}  # aggregate events in instance['events']
            for event in instance['events']:
                merged_event['type'] = event['type']
                new_event_arguments = []
                for argument in event['arguments']:
                    new_event_arguments.append(argument)
                merged_event['arguments'].extend(new_event_arguments)
            
            event_arguments = [" {}: {}".format(argument['role'], argument['name']) for
                               argument in merged_event['arguments']]
            label = " None" if not event_arguments else ";".join(event_arguments)
            return label

        # extract fs exemplars
        few_shot_list = []
        if num_few_shot_exemplars > 0:
            few_shot_insts = self._sampling_dataset(instances, 'random', num_few_shot_exemplars)
            for idx, exemplar_instance in enumerate(few_shot_insts):
                # instruction = f"ExampleText{idx+1}: " + exemplar_instance['sentence'] + ' \n' + f"ExampleAnswer{idx+1}:"
                instruction = ICL_TEXT_PREFIX[lang] + exemplar_instance['sentence'] + ' \n' + ANSWER_PREFIX[lang]
                label = extract_neweea_label(exemplar_instance)
                few_shot_list.append({
                    "id": str(idx),
                    "instruction": instruction,
                    "label": label,
                    "ground_truth": label
                })

        for idx, instance in enumerate(instances):
            if len(instance['events']) == 0:
                continue # BYzhy
            labels_str = ', '.join(labels[instance['events'][0]['type']])
            example = sample_template.copy()
            example["Samples"] = few_shot_list
            if num_few_shot_exemplars > 0:
                instruction = self._get_instruction(lang, 'newEEA', FEW_SHOT_STR)
            else:
                instruction = self._get_instruction(lang, 'newEEA')
            instruction = instruction.format(event_type=instance['events'][0]['type'], 
                                             labels_str=labels_str, text=instance['sentence']) + ANSWER_PREFIX[lang]

            label = extract_neweea_label(instance)

            example["Instance"] = {
                "id": str(idx),
                "sentence": instance['sentence'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }
            yield example
    
    def load_Common_dataset(self, lang, dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset, 
                         extra_sample_rate):
        # load common data that is not specific to any task
        with open(dataset_path,"r",encoding="utf=8") as f:
            instances = json.loads(f.read())
        
        
        sample_template = {"Language": lang, "Task": "Common", "Dataset": dataset_name, "Samples": [], "subset": subset}
        instances = self._sampling_dataset(instances, sampling_strategy, max_num_instances, extra_sample_rate)
        

        for idx,instance in enumerate(instances):
            # if dataset_name == "schedule":
            #     instance["instruction"] = "<TOKENS_UNUSED_1>" + instance["instruction"] + "<TOKENS_UNUSED_2>"
            example = sample_template.copy()
            example["Instance"] = {
                "id":str(idx),
                "sentence":"",
                "label":instance["output"],
                "ground_truth":instance["output"],
                "instruction":instance["instruction"]
            }
            yield example
        

    # This is the entry called by load_dataset! which is a generator
    def _generate_examples(self, path=None, task_config=None, max_num_instances_per_task=None, subset=None):
        print("subset",subset)
        logger.info(f"Generating tasks from = {path}")
        
        # load data for each task
        for task in task_config:
            if task.startswith('NER'):
                load_func = self.load_NER_dataset
            elif task.startswith('RE'):
                load_func = self.load_RE_dataset
            elif task.startswith('EET'):
                load_func = self.load_EET_dataset
            elif task.startswith('EEA'):
                load_func = self.load_EEA_dataset
            elif task.startswith('newEEA'):
                load_func = self.load_newEEA_dataset
            elif task.startswith('newEET'):
                load_func = self.load_newEET_dataset
            elif task.startswith('Sentiment'):
                load_func = self.load_Sentiment_dataset
            elif task == 'EE':
                load_func = self.load_EE_dataset
            elif task == 'ES':
                load_func = self.load_ES_dataset
            elif task == 'ET':
                load_func = self.load_ET_dataset
            elif task == 'EP':
                load_func = self.load_EP_dataset
            elif task == 'EPR':
                load_func = self.load_EPR_dataset
            elif task == "Common":
                load_func = self.load_Common_dataset
            else:
                raise ValueError("Unsupport {} task, plz check {} task config!".format(task, subset))

            if self.config.lang == "auto":
                lang = task.split('_')[1] if '_' in task else 'en'
            else:
                lang = self.config.lang

            # get all datasets for the task
            for dataset in task_config[task]:
                ds_name = dataset["dataset name"]

                if '_EXTRA' in task:
                    real_task_name = task.replace('_EXTRA', '')
                    ds_path = os.path.join(path, real_task_name, ds_name, subset + '.json') 
                    labels_path = os.path.join(path, real_task_name, ds_name, 'labels.json')
                elif self.config.specific_dataset and ds_name != self.config.specific_dataset:
                    continue
                else:
                    ds_path = os.path.join(path, task, ds_name, subset + '.json')  
                    labels_path = os.path.join(path, task, ds_name, 'labels.json')

                sampling_strategy = dataset.get("sampling strategy", "random")

                if task!="Common":
                    # assert os.path.exists(ds_path), "This path {} does not exist.".format(ds_path)
                    assert os.path.exists(labels_path)
                
                idx = -1
                instances = []
                # labels_path = ""
                if task == "Common":
                    for sample in load_func(lang, ds_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
                                            subset, 1):
                        idx += 1
                        instances.append(sample)
                        yield f"{task}##{ds_path}##{idx}", sample
                else:
                    if subset == "test":
                        extra_sample_rate = 1
                        num_few_shot_exemplars = self.config.num_examples_test
                        for sample in load_func(lang, ds_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
                                                subset, extra_sample_rate, num_few_shot_exemplars):
                            idx += 1
                            instances.append(sample)
                            yield f"{task}##{ds_path}##{idx}", sample

                    else:
                        # sample 0-shot first and then few-shot samples
                        num_few_shot_exemplars = 0
                        extra_sample_rate = self.config.train_0shot_prop
                        if extra_sample_rate > 0:
                            for sample in load_func(lang, ds_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
                                                    subset, extra_sample_rate, num_few_shot_exemplars):
                                idx += 1
                                instances.append(sample)
                                yield f"{task}##{ds_path}##{idx}", sample
                    
                        num_few_shot_exemplars = self.config.num_examples
                        extra_sample_rate = self.config.train_fewshot_prop
                        if extra_sample_rate > 0 and num_few_shot_exemplars > 0:
                            for sample in load_func(lang, ds_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
                                                    subset, extra_sample_rate, num_few_shot_exemplars):
                                idx += 1
                                instances.append(sample)
                                yield f"{task}##{ds_path}##{idx}", sample