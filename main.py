from rl_jailbreak.models.model import load_generator, load_target, load_reward
from trl import PPOTrainer, PPOConfig
from tqdm import tqdm      
import argparse  
import pathlib
import pandas as pd
from datasets import Dataset

def main(args):

    ppo_config = PPOConfig(
        model_name=args.generator_model,
        learning_rate=args.ppo_lr,
        batch_size=5,
    )

    generator = load_generator(args.generator_model)
    target = load_target(args.target_model)
    reward_model = load_reward(args.reward_model)

    df = pd.read_csv(args.dataset)    
    # rename the column "question" to "query"
    df = df.rename(columns={"question": "query"})
    
    # remove the other columns risk_area,types_of_harm,source,new_category
    df = df.drop(columns=["Unnamed: 0", "risk_area", "types_of_harm", "source", "new_category"])

    dataset = Dataset.from_pandas(df)
    # TODO encode dataset using target model and slice <BOS> token off

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
        dataset=dataset,
        tokenizer=generator.tokenizer,
    )
    device = ppo_trainer.accelerator.device

    # TODO: setup W&B logging

    # TODO update training loop
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        print(f"Epoch {epoch}")

        # TODO: change back to list[Tensor]
        generator_input_tokens = ppo_trainer.tokenizer(["the"] * ppo_config.batch_size, return_tensors='pt', padding='max_length', return_attention_mask=True, truncation=True, max_length=50).input_ids.to(device)
        print(generator_input_tokens)
        # print(ppo_trainer.generate(ppo_trainer.tokenizer.encode("the", return_tensors='pt')[0].to(device)))
        # TODO: change back to list[Tensor]
        generator_output_tensors = ppo_trainer.model.generate(generator_input_tokens, **generator_kwargs)
        
        batch["attack"] = ppo_trainer.tokenizer.batch_decode(generator_output_tensors)
        # print(batch["attack"])
        # print(batch["query"])
        target_inputs = [" ".join([attack, query]) for attack, query in zip(batch["attack"], batch["query"])]
        print(target_inputs)

        # TODO: convert target into pipeline object?
        target_outputs = target.generate(target_inputs)
        print(target_outputs)

        #### Compute reward score
        # TODO: check type and shape of output. HIGH PRIO
        # TODO: convert reward into pipeline object? LOW PRIO
        
        # TODO: return list of tensors
        rewards = reward_model.generate(target_outputs)
        print(rewards)

        # TODO: Add diversity metrics here

        #### Run PPO step
        # TODO: DEBUG THIS incorrect type
        stats = ppo_trainer.step(generator_input_tokens, generator_output_tensors, rewards) # Charlie: look here, see how ppo_trainer.step is doing
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

    # TODO: Add reward parameters
    ########### Target model parameters ##########
    parser.add_argument(
        "--reward-model",
        default = "nicholasKluge/ToxicityModel",
        help = "Name of reward model.",
        choices=["nicholasKluge/ToxicityModel"]
    )
    parser.add_argument(
        "--reward-max-tokens",
        type = int,
        default = 512,
        help = "Maximum number of input tokens for the reward."
    )
    ##################################################

    ########### Dataset parameters ##########
    parser.add_argument(
        "--dataset",
        default="./datasets/ppo_dataset.csv",
        help = "Path to PPO dataset.",
        type = pathlib.Path,
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
    
