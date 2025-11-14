"""Main script to run ToolACE evaluation."""

from model_wrapper import ToolACEModel
from agent import ToolACEAgent
from evaluator import ConversationEvaluator
from utils.function_parser import get_available_tools
from tools import functions as tool_functions
import config
import os
import glob


def main():
    """Main execution function."""
    # Initialize model
    print("Loading model...")
    model = ToolACEModel(config.MODEL_NAME)
    
    # Get available tools
    print("Loading tools...")
    tools = get_available_tools(tool_functions)
    print(f"Loaded {len(tools)} tools")
    print("Available tools:", [t['name'] for t in tools])
    
    # Initialize agent
    agent = ToolACEAgent(
        model=model,
        tools_module=tool_functions,
        system_prompt=config.SYSTEM_PROMPT,
        tools=tools
    )
    
    # Initialize evaluator
    # evaluator = ConversationEvaluator(agent)
    
    # Run evaluation
    # Get all JSON files from conversation_eval_set folder
    json_files = glob.glob("conversation_eval_set/*.json")
    
    if not json_files:
        print("No JSON files found in conversation_eval_set folder")
        return
    
    print(f"Found {len(json_files)} conversation files to evaluate")
    file_metrics = {}
    for input_file in json_files:
        # input_file = "conversation_dialogue_parsing.json"
        output_file = f"{input_file.split('.')[0]}_results.csv"
        evaluator = ConversationEvaluator(agent)
        print(f"\nEvaluating conversations from {input_file}...")
        metrics = evaluator.evaluate_conversation_file(input_file)

        

        # Print metrics
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Correct: {metrics['correct_predictions']}/{metrics['total_questions']}")
        print("="*60)
        
        # Save results
        evaluator.save_results(output_file)

        file_metrics[input_file] = f"{metrics['accuracy']}, {metrics['correct_predictions']}/{metrics['total_questions']}"

    # Save all metrics to a CSV file
    print(file_metrics)


if __name__ == "__main__":
    main()