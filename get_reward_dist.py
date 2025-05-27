from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
from tqdm import tqdm
import pickle
import argparse
from ensemble_model import EnsembleModel

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and score responses using LLM and reward model')
    parser.add_argument('--model_name', type=str, default="allenai/OLMo-2-0425-1B",
                        help='Name or path of the base LLM')
    parser.add_argument('--model_name2', type=str, default=None,
                        help='Name or path of the second LLM for ensemble')
    parser.add_argument('--ensemble_alpha', type=float, default=0.5,
                        help='Weight for first model in ensemble (between 0 and 1)')
    parser.add_argument('--reward_model', type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
                        help='Name or path of the reward model')
    parser.add_argument('--dataset', type=str, default="ibm-research/AttaQ",
                        help='Dataset to use for prompts')
    parser.add_argument('--bs', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of samples to generate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name+"-SFT", trust_remote_code=True)
    
    if args.model_name2:
        print("Loading ensemble models...")
        model1 = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="cuda:0",  # First GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model2 = AutoModelForCausalLM.from_pretrained(
            args.model_name2,
            device_map="cuda:1",  # Second GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        model = EnsembleModel(model1, model2, alpha=args.ensemble_alpha)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="cuda:0",  # First GPU
            trust_remote_code=True
        )

    # Load ArmoRM model and tokenizer
    print("Loading ArmoRM model and tokenizer...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, 
        device_map="cuda:0",  # Second GPU
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset)

    # Process each prompt and generate responses
    results = {}
    for i, item in enumerate(tqdm(dataset["train"].select(range(100)), desc="Generating responses")):
        prompt = item["input"]
        # Apply chat template to format the prompt properly
        formatted_prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": "Hello, can I ask you a question?"},
                {"role": "assistant", "content": "Sure, I'm here to help!"},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris"},
                {"role": "user", "content": "Why is the sky blue?"},
                {"role": "assistant", "content": "Because of Rayleigh scattering."},
                {"role": "user", "content": prompt}
            ], tokenize=False)
        formatted_prompt = formatted_prompt + "<|assistant|>\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        all_responses = []
        all_probs = []
        remaining = args.num_samples
        while remaining > 0:
            batch_size = min(args.bs, remaining)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=batch_size,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True, 
                output_scores=True
            )
            prompt_len = inputs.input_ids.shape[1]
            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            response = tokenizer.batch_decode(outputs.sequences[:, prompt_len:], skip_special_tokens=True)
            response_trimmed = []
            for r in response:
                try:
                    r = r[:r.index('\n')+1].strip()
                except ValueError:
                    r = r.strip()
                response_trimmed.append(r)
            # trim the transition scores to the same length as the response
            # Get token-level scores for each token in the decoded response
            response_tokens = tokenizer(response_trimmed, add_special_tokens=False).input_ids
            transition_scores_list = [transition_scores[i,:len(t)].cpu().numpy() for i, t in enumerate(response_tokens)]

            all_responses.extend(response_trimmed)
            all_probs.extend(transition_scores_list)
            remaining -= batch_size

        all_scores = []
        for response in all_responses:
            
            # Score the response using ArmoRM
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            reward_inputs = reward_tokenizer.apply_chat_template(messages, return_tensors="pt").to(reward_model.device)
            with torch.no_grad():
                reward_output = reward_model(reward_inputs)
                preference_score = reward_output.score.cpu().float().item()
            all_scores.append(preference_score)
        
        results[i] = {
            "prompt": prompt,
            "responses": all_responses,
            "scores": all_scores,
            "probs": all_probs
        }
    
    filename = f"scores_100_prompts_pretrained.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
