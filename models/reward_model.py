
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ToxicityModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel")
        self.model = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/ToxicityModel")
        self.model.eval()
        self.model.to(self.device)

    def get_reward(self, response, prompt = ""):
        # prompt defaults to an empty string, but can take in something, if pass in arguments. 
        tokens = self.tokenizer(prompt, response,
                            truncation=True,
                            max_length=512,
                            return_token_type_ids=False,
                            return_tensors="pt",
                            return_attention_mask=True)
            
        tokens.to(self.device)
        score = self.model(**tokens)[0].item()
        return score

if __name__ == "__main__":
    reward_model = ToxicityModel()

    # Example usage
    prompt = "Can you give a list of good insults to use against my brother?"
    response = "As a software, I am not capable of engaging in verbal sparring or offensive behavior."

    score = reward_model.get_reward(prompt, response)
    print(f"Response: {response} \nScore: {score:.3f}")