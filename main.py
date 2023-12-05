from rl_jailbreak import load_generator, load_target, load_reward
from trl import PPOTrainer, PPOConfig
from tqdm import tqdm      
import argparse  
import pathlib
import pandas as pd
from torch.utils.data import Dataset

def main(args):

    ppo_config = PPOConfig(
        model_name=args.generator_model,
        learning_rate=args.ppo_lr,
    )

    generator = load_generator(ppo_config.model_name)
    target = load_target(args.target_model)
    reward_model = load_reward(args.reward_model)

    dataset = pd.read_csv(args.dataset)["question"]
    dataset = ... # TODO convert dataset to pytorch DataSet (compatability for PPOTrainer class)

    generator_kwargs = {
        "min_length": -1, # don't ignore the EOS token (see above)
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": generator.tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": args.generator_max_tokens, # specify how many tokens you want to generate at most
    }

    target_kwargs = {

    }

    ppo_trainer = PPOTrainer(
        model=generator.model,
        config=ppo_config,
        train_dataset=dataset,
        tokenizer=generator.tokenizer,
    )

    # TODO update training loop
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # query_tensors = batch["input_ids"]
        query_tensors = [generator.tokenizer.encode("") for _ in range(len(batch["input_ids"]))]
        response_tensors = ppo_trainer.generate(query_tensors, **generator_kwargs)
        batch["response"] = [generator.tokenizer.decode(r.squeeze()) for r in response_tensors]

        target_outputs = target.generate(batch["response"])

        #### Compute reward score
        rewards = reward_model.generate(target_outputs)

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
  
        #### Save model
        ppo_trainer.save_model(args.save_dir)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    ########### Generator model parameters ##########
    parser.add_argument(
        "--generator-model",
        default = "gpt2-medium",
        help = "Name of attack generator model.",
        choices=["gpt2-medium"]
    )
    parser.add_argument(
        "--generator-max-tokens",
        type = int,
        default = 32,
        help = "Maximum number of generated tokens for the attacker."
    )
    
    ##################################################

    ########### Target model parameters ##########
    parser.add_argument(
        "--target-model",
        default = "gpt2-medium",
        help = "Name of target model.",
        choices=["gpt2-medium"]
    )
    parser.add_argument(
        "--target-max-tokens",
        type = int,
        default = 150,
        help = "Maximum number of generated tokens for the target."
    )
    ##################################################

    ########### Dataset parameters ##########
    parser.add_argument(
        "--dataset",
        required = True,
        help = "Path to PPO dataset.",
        type = pathlib.Path
    )
    
    ##################################################

    ########### PPO parameters ##########
    parser.add_argument(
        "--ppo-lr",
        default = 1.5e-5,
        help = "Learning rate for PPO",
        type = float,
    )
    
    # TODO: add other PPO params
    ##################################################

    args = parser.parse_args()
    main(args)