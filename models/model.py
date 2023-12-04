from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

class Model(object):
    def __init__(self, model, tokenizer, device) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, input):
        raise NotImplementedError()
    
class GeneratorModel(Model):
    def generate(self, input):
        input_tensor = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        return self.model.generate(input_tensor)
    
class TargetModel(Model):
    def generate(self, input):
        input_tensor = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        outputs = self.model.generate(input_tensor)
        return self.tokenizer.batch_decode(outputs)
    
class RewardModel(Model):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.reward_config = RewardConfig(

        )

    def generate(self, input):
        self.model.eval()
        tokens = self.tokenizer.encode(input, return_tensors='pt', return_attention_mask=True).to(self.device)
        return self.model(**tokens)[0].item()

    def train(self, dataset):
        trainer = RewardTrainer(
            model=self.model,
            args=self.reward_config,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            peft_config=self.peft_config,
        )
        trainer.train()