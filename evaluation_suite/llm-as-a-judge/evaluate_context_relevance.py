"""
Evaluate whether user requests in CSV file need previous context to be answered.
Uses Llama3.1-8b-instruct to classify each user message.
Output: 1 = needs context, 0 = doesn't need context
"""

import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_FILE = "outputs/faithful_merge.csv"
OUTPUT_FILE = "context_dependency_evaluation.csv"


def load_user_questions(filepath):
    """Load user questions from CSV file."""
    df = pd.read_csv(filepath)
    return df['user_question'].tolist()


def create_evaluation_prompt(user_message, previous_context=""):
    """
    Create a prompt for Llama to evaluate if the user message needs context.
    """
    prompt = f"""You are evaluating whether a user's question or request can be answered without knowing the previous conversation context.

User Message: "{user_message}"

Analyze this message and determine:
- Can this message be understood and answered completely on its own?
- Does it reference something from earlier in the conversation (like "it", "that", "the model", etc.)?
- Does it assume knowledge from previous exchanges?

Answer with ONLY a single digit:
0 = The message is self-contained and can be answered without previous context
1 = The message requires previous context to be understood or answered properly

Answer (0 or 1):"""
    
    return prompt


def get_model_response(model, tokenizer, prompt, device):
    """Get response from Llama model."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()


def extract_label(response):
    """Extract binary label from model response."""
    # Look for 0 or 1 in the response
    response = response.strip()
    if '0' in response[:5]:
        return 0
    elif '1' in response[:5]:
        return 1
    else:
        # Default to 1 (needs context) if unclear
        print(f"Warning: Unclear response '{response}', defaulting to 1")
        return 1


def evaluate_user_questions(user_questions, model, tokenizer, device):
    """
    Evaluate all user questions for context dependency.
    Returns list of (question_id, user_question, label).
    """
    results = []
    
    for idx, user_question in enumerate(tqdm(user_questions, desc="Processing questions")):
        # Skip empty or NaN questions
        if pd.isna(user_question) or str(user_question).strip() == "":
            continue
            
        # Create evaluation prompt
        prompt = create_evaluation_prompt(user_question)
        
        # Get model prediction
        response = get_model_response(model, tokenizer, prompt, device)
        label = extract_label(response)
        
        # Store result
        results.append({
            "question_id": idx,
            "user_question": user_question,
            "label": label,
            "model_response": response
        })
        
        if idx % 10 == 0:  # Print every 10 questions
            print(f"\nQuestion {idx}")
            print(f"Message: {user_question[:100]}...")
            print(f"Label: {label}")
    
    return results


def save_results(results, output_file):
    """Save results to CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'user_question', 'label', 'model_response']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_file}")


def main():
    print("Loading model and tokenizer...")
    print(f"Model: {MODEL_NAME}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    print("\nLoading user questions from CSV...")
    user_questions = load_user_questions(INPUT_FILE)
    print(f"Loaded {len(user_questions)} user questions")
    
    print("\nEvaluating context dependency...")
    results = evaluate_user_questions(user_questions, model, tokenizer, device)
    
    print(f"\nTotal user questions evaluated: {len(results)}")
    
    # Print summary statistics
    needs_context = sum(1 for r in results if r['label'] == 1)
    no_context = sum(1 for r in results if r['label'] == 0)
    print(f"Needs context (1): {needs_context} ({needs_context/len(results)*100:.1f}%)")
    print(f"No context needed (0): {no_context} ({no_context/len(results)*100:.1f}%)")
    
    # Save results
    save_results(results, OUTPUT_FILE)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
