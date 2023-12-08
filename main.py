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
from torch.utils.tensorboard import SummaryWriter


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    run_name = f"{args.model_name}-{datetime.now().strftime('%Y-%m-%d|%H:%M:%S')}"
    
    ppo_config = PPOConfig(
        model_name=args.generator_model,
        learning_rate=args.ppo_lr,
        batch_size=20,
        log_with="tensorboard",
        
        project_kwargs={
            "logging_dir":f"./logs/{run_name}",
        },
        
    )
    writer = SummaryWriter(f"./logs/{run_name}")
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
        "min_length": 100, # don't ignore the EOS token (see above)
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

    # TODO update training loop

    GLOBAL_ITER = 0
    # LOG_EVERY = 10
    for epoch in tqdm(range(args.max_epoch)):
        for batch_idx, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            
            print(f"Epoch {epoch} Batch {batch_idx}")
            
            generator_input_tokens = [ppo_trainer.tokenizer("Assign a role for a language model to perform the task: ", return_tensors='pt')['input_ids'].to(device).squeeze()] * ppo_config.batch_size
            
            generator_output_tensors = [ppo_trainer.model.generate(i.unsqueeze(0), **generator_kwargs).squeeze()[-args.max_length:] for i in generator_input_tokens]
            
            batch["attack"] = ["".join(ppo_trainer.tokenizer.batch_decode(i)) for i in generator_output_tensors]
            target_inputs = [" ".join([attack, query]) for attack, query in zip(batch["attack"], batch["query"])]
            if GLOBAL_ITER % args.log_freq == 0:
                for i, (attack, query) in enumerate(zip(batch["attack"], batch["query"])):
                    writer.add_text(f"target_input/target_input_attack[{i}]", attack, GLOBAL_ITER)
                    writer.add_text(f"target_input/target_input_query[{i}]", query, GLOBAL_ITER)
            
            target_outputs = target.generate(target_inputs)
            
            if GLOBAL_ITER % args.log_freq == 0:
                for i, val in enumerate(target_outputs):
                    writer.add_text(f"target_outputs/target_outputs[{i}]", val, GLOBAL_ITER)
            
            rewards = reward_model.generate(target_outputs)
            
            print("11111SAMPLE Target Input", 
                                        target_inputs[0], 
                                        "\n\n 11111SAMPLE ATTACK", 
                                        batch["attack"][0], 
                                        "\n\n 11111SAMPLE QUERY", 
                                        batch["query"][0], 
                                        "\n\n 11111TARGET_OUTPUT", 
                                        target_outputs[0])
            print("REWARDS_0", rewards[0])
            print("22222SAMPLE Target Input",
                                        target_inputs[1], 
                                        "\n\n 22222SAMPLE ATTACK", 
                                        batch["attack"][1], 
                                        "\n\n 22222SAMPLE QUERY", 
                                        batch["query"][1], 
                                        "\n\n 22222TARGET_OUTPUT", 
                                        target_outputs[1])
            
            print("REWARDS_0", rewards[1])
            if GLOBAL_ITER % args.log_freq == 0:
                writer.add_histogram("reward/rewards", rewards, GLOBAL_ITER)
                for i, val in enumerate(rewards):
                    writer.add_scalar(f"reward/rewards[{i}]", val, GLOBAL_ITER)
                    
            rewards = [torch.tensor([item], device=device) for item in rewards]
            # TODO: Add diversity metrics here

            #### Run PPO step
            
            stats = ppo_trainer.step(generator_input_tokens, generator_output_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
    
            # if not os.path.exists(args.save_dir):
            #     os.makedirs(args.save_dir)
            # ppo_trainer.save_model(args.save_dir)
            # generator.model.push_to_hub("my-fine-tuned-model-ppo")
            GLOBAL_ITER += 1
        writer.flush()
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
        default = "gpt2-medium", # default = "gpt2-medium",
        help = "Name of attack generator model.",
        choices=["gpt2-medium", "lvwerra/gpt2-imdb", "/data/public_models/zephyr/zephyr-7b-beta"]
    )
    parser.add_argument(
        "--generator-max-tokens",
        type = int,
        default = 150,
        help = "Maximum number of generated tokens for the attacker."
    )
    
    parser.add_argument(
        "--model_name",
        type = str,
        default = "PPO-baseline",
        help = "Model Name"
    )
    
    parser.add_argument(
        "--max_length",
        type = int,
        default = 150,
        help = "Maximum length of generated text tokens"
    )
    ##################################################

    ########### Target model parameters ##########
    parser.add_argument(
        "--target-model",
        default = "/data/public_models/zephyr/zephyr-7b-beta", # HuggingFaceH4/zephyr-7b-beta
        help = "Name of target model.",
        choices=["gpt2-medium", "lvwerra/gpt2-imdb", "/data/public_models/zephyr/zephyr-7b-beta"]
    )
    parser.add_argument(
        "--target-max-tokens",
        type = int,
        default = 150,
        help = "Maximum number of generated tokens for the target."
    )
    ##################################################

    # TODO: Add reward parameters
    ########### Reward model parameters ##########
    parser.add_argument(
        "--reward-model",
        default = "nicholasKluge/ToxicityModel",
        help = "Name of reward model.",
        choices=["nicholasKluge/ToxicityModel"]
    )
    parser.add_argument(
        "--reward-max-tokens",
        type = int,
        default = 150,
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

    parser.add_argument(
        "--ppo-num-epochs",
        default = 100,
        help = "Number of epochs for PPO",
        type = int,
    )

    parser.add_argument(
        "--ppo-batch-size",
        default = 64,
        help = "Batch size for PPO",
        type = int,
    )
    
    # TODO: add other PPO params
    ##################################################
    
    ########### Logging parameters ##########
    
    parser.add_argument(
        "--save_dir",
        default="./results",
        help = "Path to save the model.",
        type = pathlib.Path,
    )
    
    parser.add_argument(
        "--log-freq",
        default=10,
        help = "Logging frequency.",
        type = int,
    )
    

    ##################################################

    ########### Training parameters ##########
    
    parser.add_argument(
        "--max-epoch",
        default=100,
        help = "Maximum number of epochs.",
        type = int,
    )
    
    args = parser.parse_args()
    main(args)
    