from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset
from tqdm import tqdm


# Load model and datasets (we are using a test data)
# dataset = pd.read_csv("toxic_aria_test_data.csv")
dataset = pd.read_csv("toxic_aria_test_data.csv")
batch_size = 5
def process_in_batches(df, batch_size):

    # The number of batches is calculated based on the total number of rows
    num_batches = (len(df) - 1) // batch_size + 1

    for i in range(num_batches):
        # Using iloc to handle the indices and create views of the batches
        yield df.iloc[i * batch_size : min((i + 1) * batch_size, len(df))]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel")


# model_pth = "./content/ToxicityModel/checkpoint-2000"
model_name = "deberta-v3-xsmall_charlie_lr_5e-6"
base_model_pth = f"./content/{model_name}/checkpoint-"
ckpts = ["500", "1000", "1500", "2000"]
ckpts_results = []
log_file = f"{model_name}_ckpt_log.txt"  # define the name of your log file
with open(log_file, "a") as file:  # "a" means append mode
    file.write(f"Model: {model_name} Log Starts\n")
for ckpt in ckpts:
    model_pth = base_model_pth + ckpt
    toxicityModel = AutoModelForSequenceClassification.from_pretrained(model_pth)

    toxicityModel.eval()
    toxicityModel.to(device)


    softmax_sum = 0.0
    # Example of processing the DataFrame in batches
    for batch_number, batch in tqdm(enumerate(process_in_batches(dataset, batch_size), start=1)):
        # if batch_number > 1: 
        #     break
        print(f"Processing batch {batch_number}")
        # Here, perform your processing on each batch.
        # Since 'batch' is a view of the original DataFrame, changes to it will reflect in 'df'.
        print(batch["toxic_response"])
        print(batch["non_toxic_response"])
        tokens_toxic = tokenizer.batch_encode_plus(batch["toxic_response"].tolist(),
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                        return_token_type_ids=False,
                        return_tensors="pt",
                        return_attention_mask=True)
        tokens_non_toxic = tokenizer.batch_encode_plus(list(batch["non_toxic_response"]),
                        truncation=True,
                        padding="max_length",
                        max_length=512,
                        return_token_type_ids=False,
                        return_tensors="pt",
                        return_attention_mask=True)
        tokens_toxic.to(device)
        tokens_non_toxic.to(device)
        score_toxic = toxicityModel(**tokens_toxic)[0]
        score_non_toxic = toxicityModel(**tokens_non_toxic)[0]
        print(f"Score: {score_toxic}")
        print(f"Score: {score_non_toxic}")
        score_toxic = score_toxic.unsqueeze(1)  # Now shape is [batch_size, 1]
        score_non_toxic = score_non_toxic.unsqueeze(1)  # Now shape is [batch_size, 1]

        # Concatenate the tensors along the second dimension (forms pairs as rows in a matrix)
        # The shape becomes [n, 2], where n is the number of elements in each original tensor
        stacked_tensor = torch.cat((score_toxic, score_non_toxic), dim=1)

        # Apply softmax along the second dimension (across each row, i.e., across each original pair)
        softmax_result = F.softmax(stacked_tensor, dim=1)

        # The result you are interested in is the first column of the softmax result,
        # which corresponds to the results for the elements from score_toxic
        final_result = softmax_result[:, 0]
        curr_sum = final_result.sum()
        softmax_sum += curr_sum.item()
        # print("Current softmax_sum", curr_sum)
        print("final_result", softmax_sum)

    average_softmax = softmax_sum / len(dataset)
    print("AVERAGE SOFTMAX", average_softmax)
    ckpts_results.append(average_softmax)
    # Now, let's write these results to a log file.
    with open(log_file, "a") as file:  # "a" means append mode
        file.write(f"Model: {model_name}, Checkpoint: {ckpt}, Average Softmax: {average_softmax}\n")
    
print("CKPTS RESULTS", ckpts_results)
