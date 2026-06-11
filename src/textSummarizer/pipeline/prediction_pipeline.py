from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
    

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        with torch.no_grad():
            generation_output = model.generate(**inputs, **gen_kwargs)

        print("Dialogue:")
        print(text)

        output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        print("\nModel Summary:")
        print(output)

        return output 