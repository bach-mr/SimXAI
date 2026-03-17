"""Main agent logic for processing conversations.

Implements the structured response-generation algorithm:
  1. DecideToolUsage  – model decides whether a tool is needed.
  2. SelectTool / ExtractArguments – embedded in the model's tool-call output.
  3. Validate arguments; search dialogue history for missing ones, or ask the
     user for clarification (suspending the turn and resuming on the next call).
  4. CallTool  – execute the chosen tool.
  5. GenerateResponse – synthesise the tool output into a final reply.
"""

import json
import re
from typing import List, Dict, Any, Optional
from model_wrapper import ToolACEModel
from utils.function_parser import parse_function_calls
from utils.tool_executor import execute_function_call, set_conversation_messages


class ToolACEAgent:
    """Agent for processing conversations and executing tool calls."""

    def __init__(self, model: ToolACEModel, tools_module, system_prompt: str, tools: List[Dict],
                 history_window: int = None):
        """Initialise the agent.

        Args:
            model: The language model wrapper.
            tools_module: Module containing tool functions.
            system_prompt: System prompt template.
            tools: List of tool specifications.
            history_window: If set, only the last *history_window* messages
                (excluding the system message) are passed to the model at each
                generation step.  The full message list is still maintained so
                that tool results are never lost from the session state.
                None (default) means use the full history.
        """
        self.model = model
        self.tools_module = tools_module
        self.system_prompt = system_prompt
        self.tools = tools
        self.history_window = history_window
        # State kept across turns when waiting for a user clarification answer.
        self._pending_tool: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _windowed_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Return messages capped to *history_window* recent entries.

        The system message (index 0) is always included.  When history_window
        is None the full list is returned unchanged.
        """
        if self.history_window is None or len(messages) <= 1:
            return messages
        system = messages[:1]
        rest = messages[1:]
        return system + rest[-self.history_window:]

    def _find_tool_spec(self, tool_name: str) -> Optional[Dict]:
        """Return the tool specification dict for *tool_name*, or None."""
        for t in self.tools:
            if t.get('name') == tool_name:
                return t
        return None

    def _valid_args(self, args: Dict, tool_spec: Dict) -> bool:
        """Return True when all required arguments are present and non-empty."""
        required = tool_spec.get('parameters', {}).get('required', [])
        return all(k in args and args[k] not in (None, '') for k in required)

    def _find_missing_args(self, args: Dict, tool_spec: Dict) -> List[str]:
        """Return list of required argument names that are absent or empty."""
        required = tool_spec.get('parameters', {}).get('required', [])
        return [k for k in required if k not in args or args[k] in (None, '')]

    def _search_in_dialogue(self, missing: List[str], tool_name: str,
                            messages: List[Dict]) -> Dict:
        """SearchInDialogue: ask the model to extract *missing* values from history.

        Returns a dict of {arg_name: value} for any arguments that could be
        resolved from the existing conversation.  Empty dict means nothing found.
        """
        history_text = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in messages
            if m['role'] in ('user', 'assistant') and m.get('content')
        )
        prompt = (
            f"From the conversation history below, extract values for the "
            f"following arguments required by tool '{tool_name}': {missing}.\n"
            f"Respond with ONLY a JSON object containing the found key-value "
            f"pairs, or {{}} if none can be found. Do not include any other text.\n\n"
            f"Conversation history:\n{history_text}"
        )
        extraction_msgs = [
            {'role': 'system', 'content': 'You are a precise argument extractor. '
                                           'Respond with a JSON object only.'},
            {'role': 'user', 'content': prompt},
        ]
        raw = self.model.generate_response(extraction_msgs)
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            try:
                found = json.loads(match.group())
                return {k: v for k, v in found.items()
                        if k in missing and v not in (None, '')}
            except json.JSONDecodeError:
                pass
        return {}

    def _ask_user_for_clarification(self, missing: List[str], tool_name: str,
                                    messages: List[Dict]) -> str:
        """AskUserForClarification: generate a natural question for *missing* args."""
        prompt = (
            f"You need to call the tool '{tool_name}' but the following required "
            f"arguments are missing: {missing}. "
            f"Generate a single, concise, natural-language question to ask the "
            f"user for this information. Do not call any tool."
        )
        clarification_msgs = self._windowed_messages(messages) + [
            {'role': 'user', 'content': prompt}
        ]
        return self.model.generate_response(clarification_msgs)

    def _incorporate_clarification(self, args: Dict, missing: List[str],
                                   messages: List[Dict]) -> Dict:
        """UpdateArguments: parse the user's clarification reply into *args*."""
        last_user_msg = next(
            (m['content'] for m in reversed(messages) if m['role'] == 'user'), ''
        )
        prompt = (
            f"The user replied: \"{last_user_msg}\"\n"
            f"Extract values for these arguments: {missing}.\n"
            f"Respond with ONLY a JSON object of extracted key-value pairs."
        )
        extraction_msgs = [
            {'role': 'system', 'content': 'You are a precise argument extractor. '
                                           'Respond with a JSON object only.'},
            {'role': 'user', 'content': prompt},
        ]
        raw = self.model.generate_response(extraction_msgs)
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            try:
                extracted = json.loads(match.group())
                args.update({k: v for k, v in extracted.items() if k in missing})
            except json.JSONDecodeError:
                pass
        return args

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_user_message(self, messages: List[Dict[str, str]], user_message: str,
                              return_trace: bool = False) -> tuple:
        """Process a user message according to the structured generation algorithm.

        If a required tool argument cannot be resolved from history, the agent
        returns a clarification question and stores its pending state.  The next
        call to this method resumes from where it left off.

        Args:
            messages: Conversation messages (mutated in place).
            user_message: New user message.
            return_trace: If True, also return a trace dict.
        """
        messages.append({'role': 'user', 'content': user_message})

        tool_names: List[str] = []
        trace: Dict[str, List] = {"model_responses": [], "tool_payloads": []}

        # ── Resume from a pending clarification ──────────────────────────────
        if self._pending_tool:
            pending = self._pending_tool
            self._pending_tool = None
            tool_name = pending['tool']
            args = self._incorporate_clarification(pending['args'], pending['missing'],
                                                   messages)
            tool_names.append(tool_name)
        else:
            # ── Step 1: DecideToolUsage ───────────────────────────────────────
            # The model either emits a tool call (u != None) or a plain response
            # (u = None → answer from internal knowledge).
            response = self.model.generate_response(self._windowed_messages(messages))
            trace["model_responses"].append(response)

            function_calls = parse_function_calls(response)

            if not function_calls:
                # u = None: InternalKnowledge path
                final_response = response
                print("Final response:", final_response)
                messages.append({'role': 'assistant', 'content': final_response})
                if return_trace:
                    return final_response, [], messages, trace
                return final_response, [], messages

            # ── Steps 2–3: SelectTool + ExtractArguments ──────────────────────
            call = function_calls[0]
            tool_name = call['function']
            args = call['parameters']
            tool_names.append(tool_name)

        # ── Step 4: Validate arguments; fill gaps or ask for clarification ────
        tool_spec = self._find_tool_spec(tool_name)

        while tool_spec and not self._valid_args(args, tool_spec):
            missing = self._find_missing_args(args, tool_spec)

            # SearchInDialogue: try to resolve missing args from history
            found = self._search_in_dialogue(missing, tool_name, messages)
            if found:
                args.update(found)
            else:
                # AskUserForClarification: suspend turn, resume next call
                clarification_q = self._ask_user_for_clarification(
                    missing, tool_name, messages)
                self._pending_tool = {'tool': tool_name, 'args': args,
                                      'missing': missing}
                print("Clarification question:", clarification_q)
                messages.append({'role': 'assistant', 'content': clarification_q})
                if return_trace:
                    return clarification_q, tool_names, messages, trace
                return clarification_q, tool_names, messages

        # ── Step 5: CallTool ──────────────────────────────────────────────────
        set_conversation_messages(messages)
        result = execute_function_call(self.tools_module, tool_name, args)
        tool_result_str = f"{tool_name}:'{result}'"
        messages.append({"role": "tool", "name": tool_name, "content": result})
        trace["tool_payloads"].append(tool_result_str)
        print("Tool result:", tool_result_str)

        # ── Step 6: GenerateResponse ──────────────────────────────────────────
        final_response = self.model.generate_response(self._windowed_messages(messages))
        trace["model_responses"].append(final_response)
        print("Final response:", final_response)
        messages.append({'role': 'assistant', 'content': final_response})

        if return_trace:
            return final_response, tool_names, messages, trace
        return final_response, tool_names, messages

    def create_initial_messages(self) -> List[Dict[str, str]]:
        """Create initial messages with system prompt."""
        return [
            {'role': 'system', 'content': self.system_prompt.format(functions=self.tools)}
        ]
