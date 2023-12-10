from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from trl import RewardTrainer, RewardConfig, AutoModelForCausalLMWithValueHead
# from typing import Union, Tuple

def load_generator(generator_name: str):
    tokenizer = AutoTokenizer.from_pretrained(generator_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(generator_name, device_map="auto")
    return GeneratorModel(model, tokenizer)

def load_target(target_name: str):
    tokenizer = AutoTokenizer.from_pretrained(target_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(target_name, device_map="auto")
    return TargetModel(model, tokenizer)

def load_reward(reward_name: str):
    tokenizer = AutoTokenizer.from_pretrained(reward_name, padding_side='left')
    model = AutoModelForSequenceClassification.from_pretrained(reward_name, device_map="auto")
    return RewardModel(model, tokenizer)

class Model(object):
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = self.model.device
        self.model.eval()
    
    def generate(self, input):
        raise NotImplementedError()
    
class GeneratorModel(Model):
    
    def generate(self, input, generation_args={}):
        input_tensor = self.tokenizer(input, return_tensors="pt").input_ids.to(self.device)
        output_tensor = self.model.generate(input_tensor, **generation_args)
        return self.tokenizer.decode(output_tensor.squeeze(), skip_special_tokens=True)
    
class TargetModel(Model):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)

    def generate(self, input: str, generation_args={}):
        # Batch operation
        # input_tensor = self.tokenizer(input, 
        #                             return_tensors="pt", 
        #                             padding='max_length', 
        #                             return_attention_mask=True, 
        #                             truncation=True, 
        #                             max_length=256).input_ids.to(self.device)
        input_tensor = self.tokenizer(input, 
                                    return_tensors="pt")['input_ids'].to(self.device)
        # print("INPUTS", input_tensor, input_tensor.shape)
        outputs = self.model.generate(input_tensor, **generation_args)
        # print("OUTPUTS", outputs.shape)
        # print(self.tokenizer.decode(outputs.squeeze()[input_tensor.shape[1]:], skip_special_tokens=True))
        # print("FULL TEXT")
        # print(self.tokenizer.decode(outputs.squeeze(), skip_special_tokens=True))
        # print(input_tensor.shape[1])
        return self.tokenizer.decode(outputs.squeeze()[input_tensor.shape[1]:], skip_special_tokens=True)
    
class RewardModel(Model):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)
        # self.peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_CLS,
        #     inference_mode=False,
        #     r=8,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # )
        # self.reward_config = None

    def generate(self, input):
        tokens = self.tokenizer(input, return_tensors="pt", 
                                padding='max_length', 
                                return_attention_mask=True, 
                                truncation=True, 
                                max_length=512).input_ids.to(self.device)
        return -self.model(tokens).logits.squeeze() # original model predicts positive is good, negative is bad. We want more toxic to be positive, so negate. 

    # def train(self, dataset):
    #     trainer = RewardTrainer(
    #         model=self.model,
    #         args=self.reward_config,
    #         tokenizer=self.tokenizer,
    #         train_dataset=dataset,
    #         peft_config=self.peft_config,
    #     )
    #     trainer.train()
