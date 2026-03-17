"""Evaluation utilities for agent performance."""

import json
import pandas as pd
from typing import List, Dict, Any


class ConversationEvaluator:
    """Evaluate agent performance on conversations."""
    
    def __init__(self, agent):
        """Initialize evaluator."""
        self.agent = agent
        self.questions = []
        self.agent_responses = []
        self.expected_tools = []
        self.correct_count = 0
    
    def evaluate_conversation_file(self, file_path: str) -> Dict[str, Any]:
        """Evaluate conversations from a JSON file.

        Supports two JSON shapes:
          - {"dialogues": [{"conversation": [...]}, ...]}
          - {"conversations": [[...], ...]}  (list-of-lists)
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Normalise to a flat list of turn-lists
        if "dialogues" in data:
            turn_lists = [d["conversation"] for d in data["dialogues"]]
        elif "conversations" in data:
            turn_lists = data["conversations"]
        else:
            raise ValueError(
                f"Unrecognised JSON structure in {file_path}: "
                "expected top-level key 'dialogues' or 'conversations'."
            )

        for index, turns in enumerate(turn_lists):
            print(f"\n{'='*60}")
            print(f"Processing dialogue {index}")
            print('='*60)
            
            messages = self.agent.create_initial_messages()
            
            for ix, row in enumerate(turns):
                if row['speaker'] == 'user':
                    print(f"\n--- Turn {ix} ---")
                    print(f"Question: {row['message']}")
                    
                    expected_tool = row.get('tool', 'no tool')
                    self.questions.append(row['message'])
                    self.expected_tools.append(expected_tool)
                    
                    # Process message
                    response, tool_names, messages = self.agent.process_user_message(
                        messages, 
                        row['message']
                    )
                    
                    # Record results
                    agent_tool = tool_names[0] if tool_names else "no tool"
                    self.agent_responses.append(agent_tool)
                    
                    # Check correctness
                    if agent_tool in expected_tool:
                        self.correct_count += 1
                    
                    # Print results
                    print(f"Expected: {expected_tool}")
                    print(f"Agent called: {agent_tool}")
                    print(f"Response: {response}")
                    
                else:
                    messages.append({'role': 'assistant', 'content': row['message']})
        
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        accuracy = self.correct_count / len(self.questions) if self.questions else 0
        
        return {
            'accuracy': accuracy,
            'total_questions': len(self.questions),
            'correct_predictions': self.correct_count
        }
    
    def save_results(self, output_file: str):
        """Save evaluation results to CSV."""
        df = pd.DataFrame({
            "question": self.questions,
            "agent_response": self.agent_responses,
            "expected_tool": self.expected_tools
        })
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")