from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig, AutoModelForCausalLMWithValueHead

def load_generator(generator_name):
    tokenizer = AutoTokenizer.from_pretrained(generator_name)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(generator_name, device_map="auto")
    return GeneratorModel(model, tokenizer,)

def load_target(target_name):
    tokenizer = AutoTokenizer.from_pretrained(target_name)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(target_name, device_map="auto")
    return TargetModel(model, tokenizer)

def load_reward(reward_name):
    tokenizer = AutoTokenizer.from_pretrained(reward_name, device_map="auto")
    model = AutoModelForSequenceClassification.from_pretrained(reward_name)
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
        input_tensor = self.tokenizer.encode(input, return_tensors="pt")
        return self.model.generate(input_tensor)
    
class TargetModel(Model):
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)
        self.model.eval()

    def generate(self, input):
        input_tensor = self.tokenizer.encode(input, return_tensors="pt")
        outputs = self.model.generate(input_tensor)
        return self.tokenizer.batch_decode(outputs)
    
class RewardModel(Model):
<<<<<<< HEAD
        # self.peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_CLS,
        #     inference_mode=False,
        #     r=8,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        # )
        # self.reward_config = None
    def __init__(self, model, tokenizer) -> None:
        super().__init__(model, tokenizer)
        self.model.eval()

    def generate(self, input):
        tokens = self.tokenizer.encode(input, return_tensors='pt', return_attention_mask=True)
        return self.model(**tokens)[0].item()

    # def train(self, dataset):
    #     trainer = RewardTrainer(
    #         model=self.model,
    #         args=self.reward_config,
    #         tokenizer=self.tokenizer,
    #         train_dataset=dataset,
    #         peft_config=self.peft_config,
    #     )
    #     trainer.train()
=======
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.reward_config = None

    def generate(self, input):
        self.model.eval()
        tokens = self.tokenizer.encode(input, 
                    truncation=True,
                    max_length=512,
                    return_token_type_ids=False,
                    return_tensors='pt', 
                    return_attention_mask=True)
        return self.model(**tokens)[0].item()


    # This doesn't get called for now, so dead code
    def train(self, dataset):
        trainer = RewardTrainer(
            model=self.model,
            args=self.reward_config,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            peft_config=self.peft_config,
        )
        trainer.train()
>>>>>>> origin/main
