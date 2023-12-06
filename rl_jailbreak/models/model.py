from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from trl import RewardTrainer, RewardConfig, AutoModelForCausalLMWithValueHead
from typing import Union, Tuple

def load_generator(generator_name):
    tokenizer = AutoTokenizer.from_pretrained(generator_name)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(generator_name, device_map="auto")
    return GeneratorModel(model, tokenizer,)

def load_target(target_name):
    tokenizer = AutoTokenizer.from_pretrained(target_name)
    model = AutoModelForCausalLM.from_pretrained(target_name, device_map="auto")
    return TargetModel(model, tokenizer)

def load_reward(reward_name):
    tokenizer = AutoTokenizer.from_pretrained(reward_name)
    model = AutoModelForSequenceClassification.from_pretrained(reward_name, device_map="auto")
    return RewardModel(model, tokenizer)

class Model(object):
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, input):
        raise NotImplementedError()
    
class GeneratorModel(Model):
    def generate(self, input):
        input_tensor = self.tokenizer(input, return_tensors="pt")
        return self.model.generate(input_tensor)
    
class TargetModel(Model):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)
        self.model.eval()
        self.device = self.model.device

    def generate(self, input: list[str]):
        input_tensor = self.tokenizer.encode(input, return_tensors="pt")
        # input_tensor = [self.tokenizer.encode(i, return_tensors="pt") for i in input]
        # input_tensor = self.tokenizer(input, return_tensors="pt", padding='max_length', return_attention_mask=True, truncation=True, max_length=50).input_ids.to(self.device)
        outputs = self.model.generate(input_tensor)
        return self.tokenizer.batch_decode(outputs)[0]
    
class RewardModel(Model):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)
        self.model.eval()
        self.device = self.model.device
        # self.peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_CLS,
        #     inference_mode=False,
        #     r=8,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # )
        # self.reward_config = None

    def generate(self, input):
        tokens = self.tokenizer.encode(input, 
                    truncation=True,
                    max_length=512,
                    return_token_type_ids=False,
                    return_tensors='pt', 
                    return_attention_mask=True)
        # tokens = self.tokenizer(input, return_tensors='pt', return_attention_mask=True, padding='max_length', truncation=True, max_length=50).input_ids.to(self.device)
        return self.model(tokens).loss['logits'][0][0]

    # def train(self, dataset):
    #     trainer = RewardTrainer(
    #         model=self.model,
    #         args=self.reward_config,
    #         tokenizer=self.tokenizer,
    #         train_dataset=dataset,
    #         peft_config=self.peft_config,
    #     )
    #     trainer.train()
