import os
import pickle
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate responses with an additional reward model')
    parser.add_argument('--input', type=str, default="harmlessness_scores_pretrained.pkl",
                        help='Path to input pickle file from evaluate_llm_as_judge.py')
    parser.add_argument('--output', type=str, default="multi_reward_scores_pretrained_3.pkl",
                        help='Path to output pickle file with additional reward scores')
    parser.add_argument('--reward_model', type=str, default="PKU-Alignment/beaver-7b-v1.0-reward",
                        help='Name or path of the additional reward model to use')
    parser.add_argument('--reward_name', type=str, default="additional_reward",
                        help='Name for the additional reward metric in the output')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for reward model inference')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help='Device to run the reward model on')
    return parser.parse_args()

def load_results(path):
    """Load the pickle file containing the responses and scores."""
    logging.info(f"Loading results from {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    args = parse_args()
    
    # Load enhanced results from evaluate_llm_as_judge.py
    results = load_results(args.input)
    
    # Load additional reward model and tokenizer
    logging.info(f"Loading reward model: {args.reward_model}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        device_map=args.device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)
    
    error_in_chat_template = False
    # Process each prompt and its responses
    for idx, item in tqdm(results.items(), desc="Evaluating with additional reward model"):
        prompt = item["prompt"]
        responses = item["responses"]
        
        # Create batches for more efficient processing
        all_scores = []
        
        # Process responses in batches
        for i in range(0, len(responses), args.batch_size):
            batch_responses = responses[i:i + args.batch_size]
            batch_inputs = []
            
            # Prepare inputs for each response in the batch
            for response in batch_responses:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                try:
                    # Try using the chat template first
                    encoded = reward_tokenizer.apply_chat_template(
                        messages, 
                        return_tensors="pt",
                    ).to(reward_model.device)
                    batch_inputs.append(encoded)
                except Exception as e:
                    # Fall back to simple concatenation if chat template fails
                    if not error_in_chat_template:
                        logging.warning(f"Chat template failed: {e}. Using simple concatenation.")
                        error_in_chat_template = True
                    text = f"BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT: {response}"
                    encoded = reward_tokenizer(
                        text,
                        return_tensors="pt"
                    ).to(reward_model.device)
                    batch_inputs.append(encoded)
            
            # Process each input individually (since they may have different lengths)
            batch_scores = []
            for input_tensor in batch_inputs:
                with torch.no_grad():
                    try:
                        reward_output = reward_model(**input_tensor)
                        # Different reward models may have different output formats
                        if hasattr(reward_output, 'score'):
                            score = reward_output.score.cpu().float().item()
                        else:
                            # Handle logits-based output
                            logits = reward_output.logits
                            if logits.dim() > 1 and logits.size(1) > 1:
                                # For models that output multiple scores, take the positive class
                                score = torch.softmax(logits, dim=1)[:, 1].cpu().float().item()
                            else:
                                # For models with a single score
                                score = logits.cpu().float().item()
                        batch_scores.append(score)
                    except Exception as e:
                        logging.error(f"Error scoring response: {e}")
                        batch_scores.append(float('nan'))  # Use NaN to indicate scoring failure
            
            all_scores.extend(batch_scores)
        
        # Add the additional reward scores to the results dictionary
        item[args.reward_name + "_scores"] = all_scores
    
    # Save enhanced results
    logging.info(f"Saving enhanced results to {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    
    # Print some statistics
    total_responses = sum(len(item["responses"]) for item in results.values())
    avg_score = sum(sum(s for s in item[args.reward_name + "_scores"] if not torch.isnan(torch.tensor(s))) 
                  for item in results.values()) / total_responses
    
    logging.info(f"Evaluation complete!")
    logging.info(f"Total responses evaluated: {total_responses}")
    logging.info(f"Average {args.reward_name} score: {avg_score:.4f}")

if __name__ == "__main__":
    main() 
