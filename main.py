from rl_jailbreak.models.model import load_generator, load_target, load_reward
from trl import PPOTrainer, PPOConfig
from tqdm import tqdm      
import argparse  
import pathlib
import pandas as pd
from datasets import Dataset
import os 
import torch
from datetime import datetime
import wandb

wandb.init() # initialize W&B project

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    ppo_config = PPOConfig(
        model_name=args.generator_model,
        learning_rate=args.ppo_lr,
        batch_size=5,
        log_with="wandb",
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
    MAX_EPOCH = 100
    for epoch in tqdm(range(MAX_EPOCH)):
        for batch_idx, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            print(f"Epoch {epoch} Batch {batch_idx}")
            # TODO: change back to list[Tensor]
            MAX_LENGTH = 50
            # generator_input_tokens = [ppo_trainer.tokenizer("the" , return_tensors='pt')['input_ids'].to(device).squeeze()] * ppo_config.batch_size
            generator_input_tokens = [ppo_trainer.tokenizer("the" , return_tensors='pt', padding='max_length', return_attention_mask=True, truncation=True, max_length=MAX_LENGTH)['input_ids'].to(device).squeeze()] * ppo_config.batch_size
            print(generator_input_tokens)
            # print(ppo_trainer.generate(ppo_trainer.tokenizer.encode("the", return_tensors='pt')[0].to(device)))
            # TODO: change back to list[Tensor]
            generator_output_tensors = [ppo_trainer.model.generate(generator_input_tokens[0].unsqueeze(0), **generator_kwargs).squeeze()[-MAX_LENGTH:] for i in generator_input_tokens]
            batch["attack"] = [ppo_trainer.tokenizer.batch_decode(i)[0] for i in generator_output_tensors]
            target_inputs = [" ".join([attack, query]) for attack, query in zip(batch["attack"], batch["query"])]
            print(target_inputs)

            # TODO: convert target into pipeline object?
            # target_outputs = [target.generate(i) for i in target_inputs]
            target_outputs = target.generate(target_inputs)
            # print(target_outputs)

            #### Compute reward score
            # TODO: check type and shape of output. HIGH PRIO
            # TODO: convert reward into pipeline object? LOW PRIO
            
            # TODO: return list of tensors
            # rewards = [reward_model.generate(i) for i in target_outputs]
            rewards = reward_model.generate(target_outputs)
            # print(rewards)
            rewards = [torch.tensor([item], device=device) for item in rewards]
            # TODO: Add diversity metrics here

            #### Run PPO step
            # TODO: DEBUG THIS incorrect type
            stats = ppo_trainer.step(generator_input_tokens, generator_output_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
    
            #### Save model
            # FIXME: @Xavier the following three lines.
            """
            Traceback (most recent call last):
            File "/data1/charlieji/rl_jailbreak/main.py", line 170, in <module>
                main(args)
            File "/data1/charlieji/rl_jailbreak/main.py", line 94, in main
                if not os.path.exists(args.save_dir):
            AttributeError: 'Namespace' object has no attribute 'save_dir'
            """
            # if not os.path.exists(args.save_dir):
            #     os.makedirs(args.save_dir)
            # ppo_trainer.save_model(args.save_dir)
            # generator.model.push_to_hub("my-fine-tuned-model-ppo")
        if epoch % 5 == 0:
            # ppo_trainer.save_model(args.save_dir)
            # get current time
            pass
            # ppo_trainer.save_pretrained(f"{args.save_dir}/{args.model_name}{datetime.now().strftime("%Y-%m-%d|%H:%M:%S")}-EPOCH-{epoch}") 
            # print(f"Model saved at {args.save_dir}/{datetime.now().strftime("%Y-%m-%d|%H:%M:%S")}-EPOCH-{epoch}")
            

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
    
    parser.add_argument(
        "--model_name",
        type = str,
        default = "PPO-baseline",
        help = "Model Name"
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
    
    parser.add_argument(
        "--save_dir",
        default="./results",
        help = "Path to save the model.",
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
    
