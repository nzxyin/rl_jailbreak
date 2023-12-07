from rl_jailbreak.models.model import load_generator, load_target, load_reward
from trl import SFTTrainer
from transformers import AutoModelForCausalLM
from tqdm import tqdm      
import argparse  
import pathlib
import pandas as pd
from datasets import Dataset
import os 
from peft import LoraConfig


def main(args):

    # Handle Logging
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    

    df = pd.read_csv(args.dataset)    
    # rename the column "question" to "query"
    df = df.rename(columns={"question": "query"})
    
    # remove the other columns risk_area,types_of_harm,source,new_category
    df = df.drop(columns=["Unnamed: 0", "risk_area", "types_of_harm", "source", "new_category"])

    dataset = Dataset.from_pandas(df)
    # TODO encode dataset using target model and slice <BOS> token off

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(args.generator_model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text", # TODO: header name here
        peft_config=peft_config, 
        max_seq_length=512,
    )

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
        if epoch >= args.ppo_num_epochs:
            break

        print(f"Epoch {epoch}")

        # TODO: change back to list[Tensor]
        MAX_LENGTH = args.ppo_
        generator_input_tokens = [ppo_trainer.tokenizer("the", return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LENGTH).input_ids.to(device).squeeze()] * ppo_config.batch_size
        print(generator_input_tokens)
        # print(ppo_trainer.generate(ppo_trainer.tokenizer.encode("the", return_tensors='pt')[0].to(device)))
        # TODO: change back to list[Tensor]
        # ppo_trainer.model.generate(i, **generator_kwargs)
        generator_output_tensors = [ppo_trainer.model.generate(generator_input_tokens[0].unsqueeze(0), **generator_kwargs).squeeze()[-MAX_LENGTH:] for i in generator_input_tokens]
        
        batch["attack"] = [ppo_trainer.tokenizer.batch_decode(i)[0] for i in generator_output_tensors]
        # print(batch["attack"])
        # print(batch["query"])
        target_inputs = [" ".join([attack, query]) for attack, query in zip(batch["attack"], batch["query"])]
        print(target_inputs)

        # TODO: convert target into pipeline object?
        target_outputs = [target.generate(i) for i in target_inputs]
        print(target_outputs)

        #### Compute reward score
        # TODO: check type and shape of output. HIGH PRIO
        # TODO: convert reward into pipeline object? LOW PRIO
        
        # TODO: return list of tensors
        rewards = [reward_model.generate(i) for i in target_outputs]
        print(rewards)

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
        # 
        ppo_trainer.save_model(args.save_dir)
        # generator.model.push_to_hub("my-fine-tuned-model-ppo")
        # generator.model.save_pretrained("my-fine-tuned-model-ppo") # source: https://huggingface.co/docs/trl/quickstart, only the most recent one got saved. FIXME


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

    ########### Dataset parameters ##########
    parser.add_argument(
        "--dataset",
        default="./datasets/sft_dataset.csv",
        help = "Path to SFT dataset.",
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

    ##################################################

    args = parser.parse_args()
    main(args)
    
