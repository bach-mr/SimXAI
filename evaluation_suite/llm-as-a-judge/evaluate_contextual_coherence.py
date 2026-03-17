"""
Evaluate contextual coherence of agent responses using conversation history.

This script:
1. Reads context_annotated.csv filtered for rows with context_relevant == 1
2. Reconstructs full conversation history from source JSON files
3. Uses Llama-3.1-8B-Instruct as LLM-as-a-judge to evaluate if agent_response is:
   - Logically unified (internally consistent and well-structured)
   - Semantically relevant to the user_question
   - Consistent with established conversation history
4. Outputs results with coherence labels and explanations
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_CSV = "evaluation/dependency_context.csv"
OUTPUT_CSV = "evaluation/dependency_context_results.csv"

# JSON dialogue files to search
DIALOGUE_FILES = [
    "dialogues_HR.json",
    "dialogues_HR_2.json",
    "dialogues_SA.json"
]


def load_dialogue_files() -> Dict[str, List[List[Dict]]]:
    """Load all dialogue JSON files."""
    dialogues = {}
    for filename in DIALOGUE_FILES:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                dialogues[filename] = json.load(f)
            print(f"Loaded {filename}: {len(dialogues[filename])} dialogues")
    return dialogues


def find_conversation_context(
    user_question: str,
    agent_response: str,
    dialogues: Dict[str, List[List[Dict]]]
) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """
    Find the conversation history for a given user_question and agent_response.
    
    Returns:
        Tuple of (source_file, conversation_history) where conversation_history
        includes all turns up to (but not including) the current agent response.
    """
    for filename, dialogue_list in dialogues.items():
        for dialogue in dialogue_list:
            # Search through turns to find matching user_question
            for i, turn in enumerate(dialogue):
                if turn.get('role') in ['User', 'user'] and turn.get('message', '') == user_question:
                    # Check if the next turn (agent response) matches
                    if i + 1 < len(dialogue):
                        next_turn = dialogue[i + 1]
                        if next_turn.get('role') in ['Explainer', 'explainer', 'Agent', 'agent']:
                            # Found the match - return history up to this point
                            history = dialogue[:i]
                            return filename, history
    
    # If not found, return empty history
    return None, []


def format_conversation_history(history: List[Dict]) -> str:
    """Format conversation history as a readable string."""
    if not history:
        return "No previous conversation context."
    
    formatted = []
    for turn in history:
        role = turn.get('role', 'Unknown')
        message = turn.get('message', '')
        formatted.append(f"{role}: {message}")
    
    return "\n".join(formatted)


def create_coherence_evaluation_prompt(
    conversation_history: str,
    user_question: str,
    agent_response: str
) -> str:
    """
    Create a prompt for Llama to evaluate contextual coherence.
    
    The LLM judges whether the agent_response is:
    - Logically unified (internally consistent and well-structured)
    - Semantically relevant to the user_question
    - Consistent with the established conversation history
    """
    prompt = f"""You are evaluating the contextual coherence of an AI agent's response in a conversation.

CONVERSATION HISTORY:
{conversation_history}

CURRENT USER QUESTION:
{user_question}

AGENT RESPONSE:
{agent_response}

Evaluate whether the AGENT RESPONSE is contextually coherent, which means:
1. Logically unified: The response is internally consistent and well-structured
2. Semantically relevant: The response directly addresses the user's question
3. Historically consistent: The response is consistent with information from the conversation history

You must respond in strict JSON format with NO additional text:
{{
  "coherent": true,
  "explanation": "Brief reasoning in 1-3 sentences"
}}

OR

{{
  "coherent": false,
  "explanation": "Brief reasoning in 1-3 sentences"
}}

Your JSON response:"""
    
    return prompt


def get_model_response(model, tokenizer, prompt: str, device) -> str:
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
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()


def parse_json_response(response: str) -> Tuple[bool, str]:
    """
    Parse JSON response from the model.
    
    Returns:
        Tuple of (coherent: bool, explanation: str)
    """
    try:
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle case where response might have extra text
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx+1]
            result = json.loads(json_str)
            
            coherent = result.get('coherent', False)
            explanation = result.get('explanation', 'No explanation provided')
            
            return coherent, explanation
        else:
            # Fallback: try to infer from text
            coherent = 'true' in response.lower() or 'coherent' in response.lower()
            return coherent, response
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response}")
        # Fallback
        coherent = 'true' in response.lower() or 'coherent' in response.lower()
        return coherent, response
    except Exception as e:
        print(f"Unexpected error parsing response: {e}")
        return False, f"Error: {str(e)}"


def evaluate_coherence(
    df: pd.DataFrame,
    dialogues: Dict[str, List[List[Dict]]],
    model,
    tokenizer,
    device
) -> List[Dict]:
    """
    Evaluate contextual coherence for all rows with context_relevant == 1.
    
    Returns:
        List of result dictionaries with evaluation data
    """
    results = []
    
    # Filter for context-relevant rows
    context_relevant_df = df[df['context relevant'] == 1].copy()
    
    print(f"\nEvaluating {len(context_relevant_df)} context-relevant rows...")
    
    for idx, row in tqdm(context_relevant_df.iterrows(), total=len(context_relevant_df)):
        user_question = row['user_question']
        agent_response = row['agent_response']
        
        # Find conversation context
        source_file, history = find_conversation_context(
            user_question, agent_response, dialogues
        )
        
        # Format history
        formatted_history = format_conversation_history(history)
        
        # Create prompt
        prompt = create_coherence_evaluation_prompt(
            formatted_history,
            user_question,
            agent_response
        )
        
        # Get model evaluation
        try:
            model_response = get_model_response(model, tokenizer, prompt, device)
            coherent, explanation = parse_json_response(model_response)
        except Exception as e:
            print(f"\nError evaluating row {idx}: {e}")
            coherent = False
            explanation = f"Evaluation error: {str(e)}"
            model_response = ""
        
        # Store results
        result = {
            'row_index': idx,
            'user_question': user_question,
            'agent_response': agent_response,
            'source_file': source_file or 'not_found',
            'conversation_turns_before': len(history),
            'coherent': coherent,
            'explanation': explanation,
            'raw_model_response': model_response
        }
        
        results.append(result)
    
    return results


def save_results(results: List[Dict], output_file: str):
    """Save evaluation results to CSV."""
    if not results:
        print("No results to save!")
        return
    
    fieldnames = [
        'row_index',
        'user_question',
        'agent_response',
        'source_file',
        'conversation_turns_before',
        'coherent',
        'explanation',
        'raw_model_response'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    coherent_count = sum(1 for r in results if r['coherent'])
    total = len(results)
    print(f"\nSummary:")
    print(f"  Total evaluated: {total}")
    print(f"  Coherent: {coherent_count} ({coherent_count/total*100:.1f}%)")
    print(f"  Not coherent: {total - coherent_count} ({(total - coherent_count)/total*100:.1f}%)")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Contextual Coherence Evaluation")
    print("=" * 60)
    
    # Load CSV data
    print(f"\nLoading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} total rows")
    print(f"Rows with context_relevant == 1: {(df['context relevant'] == 1).sum()}")
    
    # Load dialogue files
    print("\nLoading dialogue files...")
    dialogues = load_dialogue_files()
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    print("This may take a few minutes...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    
    print("Model loaded successfully!")
    
    # Evaluate coherence
    results = evaluate_coherence(df, dialogues, model, tokenizer, device)
    
    # Save results
    save_results(results, OUTPUT_CSV)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
