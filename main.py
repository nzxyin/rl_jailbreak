from model import GeneratorModel, TargetModel, RewardModel
from trl import PPOTrainer, PPOConfig
from tqdm import tqdm

class AttackTrainer(object):
    def __init__(self,
                 generator,
                 target,
                 reward_model,
                 device,
                 save_dir,
                 generator_name,
                 lr):
        self.generator = generator
        self.target = target
        self.reward_model = reward_model
        self.device = device
        self.save_dir = save_dir

        self.config = PPOConfig(
            model_name=generator_name,
            learning_rate=lr,
        )

        self.generation_kwargs = {
            "min_length": -1, # don't ignore the EOS token (see above)
            "top_k": 0.0, # no top-k sampling
            "top_p": 1.0, # no nucleus sampling
            "do_sample": True, # yes, we want to sample
            "pad_token_id": self.generator.tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
            "max_new_tokens": 32, # specify how many tokens you want to generate at most
        }

    def train(self, dataset):

        ppo_trainer = PPOTrainer(
            model=self.generator.model,
            config=self.config,
            train_dataset=dataset,
            tokenizer=self.generator.tokenizer,
        )

        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            # query_tensors = batch["input_ids"]
            query_tensors = [self.generator.tokenizer.encode("") for _ in range(len(batch["input_ids"]))]
            response_tensors = ppo_trainer.generate(query_tensors, **self.generation_kwargs)
            batch["response"] = [self.generator.tokenizer.decode(r.squeeze()) for r in response_tensors]

            target_outputs = self.target.generate(batch["response"])

            #### Compute reward score
            rewards = self.reward_model.generate(target_outputs)

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        #### Save model
        ppo_trainer.save_model(self.save_dir)
