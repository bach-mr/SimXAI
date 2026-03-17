"""Orchestrator for running conversations between two LLM agents."""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from agent import ToolACEAgent
from model_wrapper import ToolACEModel
from user_simulator import UserSimulator
from config import MODEL_NAME, SYSTEM_PROMPT
from tools import functions
from utils.function_parser import get_available_tools


class AgentConversationOrchestrator:
    """Manages conversations between user simulator and ToolACE agent."""
    
    def __init__(self, toolace_agent: ToolACEAgent, user_simulator: UserSimulator, 
                 max_turns: int = 10, save_dir: str = "agent_conversations"):
        """Initialize the orchestrator.
        
        Args:
            toolace_agent: The ToolACE agent that answers questions
            user_simulator: The user simulator that asks questions
            max_turns: Maximum number of conversation turns
            save_dir: Directory to save conversation logs
        """
        self.toolace_agent = toolace_agent
        self.user_simulator = user_simulator
        self.max_turns = max_turns
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.full_conversation = []
        self.toolace_messages = []
    
    def run_conversation(self, initial_greeting: str = None) -> List[Dict]:
        """Run a full conversation between the two agents.
        
        Args:
            initial_greeting: Optional initial message from the ToolACE agent
            
        Returns:
            List of conversation turns with metadata
        """
        print("=" * 80)
        print("Starting Agent-to-Agent Conversation")
        print("=" * 80)
        
        # Initialize ToolACE messages
        self.toolace_messages = self.toolace_agent.create_initial_messages()
        
        # Send initial greeting if provided
        # if initial_greeting:
        #     print(f"\n🤖 ToolACE Agent: {initial_greeting}\n")
        #     self.full_conversation.append({
        #         "turn": 0,
        #         "speaker": "toolace_agent",
        #         "message": initial_greeting,
        #         "tools_used": None,
        #         "timestamp": datetime.now().isoformat()
        #     })
            
        #     # Let user simulator see the greeting
        #     _ = self.user_simulator.generate_next_question(initial_greeting)
        
        # Run conversation turns
        response = ""
        for turn in range(1, self.max_turns + 1):
            print(f"\n--- Turn {turn}/{self.max_turns} ---")
            
            # User simulator asks a question
            if turn == 1:
                user_question = self.user_simulator.generate_next_question(initial_greeting)
            else:
                user_question = self.user_simulator.generate_next_question(response)
            print(f"\n👤 User Simulator: {user_question}")
            
            self.full_conversation.append({
                "turn": turn,
                "speaker": "user_simulator",
                "message": user_question,
                "tools_used": None,
                "timestamp": datetime.now().isoformat()
            })
            
            # ToolACE agent responds
            try:
                response, tools_used, self.toolace_messages = self.toolace_agent.process_user_message(
                    self.toolace_messages, 
                    user_question
                )
                
                print(f"\n🤖 ToolACE Agent: {response}")
                if tools_used:
                    print(f"   (Used tools: {', '.join(tools_used)})")
                
                self.full_conversation.append({
                    "turn": turn,
                    "speaker": "toolace_agent",
                    "message": response,
                    "tools_used": tools_used,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                print(f"\n❌ Error: {error_msg}")
                self.full_conversation.append({
                    "turn": turn,
                    "speaker": "toolace_agent",
                    "message": error_msg,
                    "tools_used": None,
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                })
                break
        
        print("\n" + "=" * 80)
        print("Conversation Complete")
        print("=" * 80)
        
        return self.full_conversation
    
    def save_conversation(self, filename: str = None) -> str:
        """Save the conversation to a JSON file.
        
        Args:
            filename: Optional filename. If not provided, generates timestamp-based name.
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_conversation_{timestamp}.json"
        
        filepath = self.save_dir / filename
        
        conversation_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_turns": len([c for c in self.full_conversation if c["speaker"] == "user_simulator"]),
                "model_name": MODEL_NAME,
                "max_turns": self.max_turns
            },
            "conversation": self.full_conversation
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Conversation saved to: {filepath}")
        return str(filepath)
    
    def get_conversation_summary(self) -> Dict:
        """Get a summary of the conversation."""
        user_turns = [c for c in self.full_conversation if c["speaker"] == "user_simulator"]
        agent_turns = [c for c in self.full_conversation if c["speaker"] == "toolace_agent"]
        tools_used = [tool for turn in agent_turns if turn.get("tools_used") for tool in turn["tools_used"]]
        
        return {
            "total_turns": len(user_turns),
            "user_messages": len(user_turns),
            "agent_messages": len(agent_turns),
            "tools_called": len(tools_used),
            "unique_tools": list(set(tools_used)) if tools_used else [],
            "errors": len([c for c in self.full_conversation if c.get("error", False)])
        }


def main():
    """Run a sample agent-to-agent conversation."""
    print("Initializing agents...")
    
    # Initialize ToolACE agent
    print("Loading ToolACE model...")
    model = ToolACEModel(MODEL_NAME)
    tools = get_available_tools(functions)
    toolace_agent = ToolACEAgent(model, functions, SYSTEM_PROMPT, tools)
    
    # Initialize user simulator
    print("Initializing user simulator...")
    user_simulator = UserSimulator()
    
    # Create orchestrator
    orchestrator = AgentConversationOrchestrator(
        toolace_agent=toolace_agent,
        user_simulator=user_simulator,
        max_turns=8  # Can adjust this
    )
    
    # Run conversation with initial greeting
    # initial_greeting = (
    #     "Welcome to the lottery! I'm here to help you understand the model that determines the prize for your ticket. "
    #     "Each ticket contains three numbers (from 1 to 9), separated by commas, for example: 1,2,3.\n"
    #     "Your task is to explore how the model works by talking with me. "
    #     "I will help you retrieve and understand information from the model. "
    #     "You can start with providing your ticket by entering three numbers separated by commas."
    # )
    initial_greeting = (
        "Welcome to the Heart Rate Monitor! I'm here to help you understand how heart rate (BPM) affects health status. "
        "You can start by providing your heart rate in beats per minute (BPM), and I will help you understand what it means."
    )
    conversation = orchestrator.run_conversation(initial_greeting=initial_greeting)
    
    # Save conversation
    filepath = orchestrator.save_conversation()
    
    # Print summary
    summary = orchestrator.get_conversation_summary()
    print("\n" + "=" * 80)
    print("Conversation Summary:")
    print("=" * 80)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return filepath


if __name__ == "__main__":
    main()
