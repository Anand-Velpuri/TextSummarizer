import os
from src.textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk

from src.textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
    

    def convert_examples_to_features(self, example_batch):
        # Tokenize both inputs and targets in a single call
        model_inputs = self.tokenizer(
            example_batch['dialogue'],
            text_target=example_batch['summary'],
            max_length=1024,
            max_target_length=128,
            truncation=True
        )
        return model_inputs
    
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
