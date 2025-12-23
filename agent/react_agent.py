"""
ReAct Agent implementation for autonomous coding tasks.

This agent uses the ReAct (Reasoning + Acting) pattern to iteratively
plan, execute, and verify actions until a task is complete.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import litellm
from litellm import completion

from env.sandbox import Sandbox
from env.tools import Tools


class ReActAgent:
    """
    ReAct Agent that autonomously completes coding tasks using tools.

    The agent follows a loop:
    1. Receive task description
    2. Think about next action
    3. Call a tool (or finish)
    4. Observe result
    5. Repeat until done or max steps reached
    """

    def __init__(
        self,
        sandbox: Sandbox,
        model_name: str = "gpt-4o-mini",
        max_steps: int = 50,
        verbose: bool = True,
        agent_log_path: Optional[Path] = None,
        system_log_path: Optional[Path] = None
    ):
        """
        Initialize the ReAct agent.

        Args:
            sandbox: Sandbox instance for isolated execution
            model_name: LiteLLM model identifier
            max_steps: Maximum number of steps before stopping
            verbose: Whether to print agent actions
            agent_log_path: Path to agent.log for tool call logging
            system_log_path: Path to system.log for command output logging
        """
        self.sandbox = sandbox
        self.model_name = model_name
        self.max_steps = max_steps
        self.verbose = verbose

        # Initialize tools with log paths
        self.tools = Tools(
            sandbox,
            agent_log_path=agent_log_path,
            system_log_path=system_log_path
        )

        # Load prompts and schemas
        prompts_dir = Path(__file__).parent / "prompts"
        self.system_prompt = self._load_system_prompt(prompts_dir / "system.txt")
        self.tool_schema = self._load_tool_schema(prompts_dir / "tool_schema.json")

        # Create tool map linking schema names to tool methods
        self.tool_map: Dict[str, Callable] = {
            "write_file": self.tools.write_file,
            "read_file": self.tools.read_file,
            "run_command": self.tools.run_command,
            "install_deps": self.tools.install_deps,
            "start_server": self.tools.start_server,
            "finish_task": self.tools.finish_task,
        }

        # State
        self.messages: List[Dict[str, Any]] = []
        self.is_done = False
        self.step_count = 0

    def _load_system_prompt(self, path: Path) -> str:
        """Load system prompt from file."""
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found: {path}")
        return path.read_text(encoding="utf-8")

    def _load_tool_schema(self, path: Path) -> List[Dict[str, Any]]:
        """Load tool schema from JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Tool schema not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def _log(self, message: str, prefix: str = "→") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"{prefix} {message}")

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Dictionary with execution result
        """
        if tool_name not in self.tool_map:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}. Available: {list(self.tool_map.keys())}"
            }

        try:
            # Get the tool function
            tool_func = self.tool_map[tool_name]

            # Call the tool with arguments
            result = tool_func(**arguments)

            # Convert ToolResult to dict
            return result.to_dict()

        except TypeError as e:
            return {
                "success": False,
                "error": f"Invalid arguments for {tool_name}: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing {tool_name}: {str(e)}"
            }

    def run(self, task: str) -> Dict[str, Any]:
        """
        Run the agent on a task.

        Args:
            task: Task description/prompt

        Returns:
            Dictionary with run results (success, steps, final_state, etc.)
        """
        # Initialize messages with system prompt and task
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        self.is_done = False
        self.step_count = 0

        self._log(f"Starting task: {task}", prefix="🎯")
        self._log(f"Max steps: {self.max_steps}", prefix="⚙️")

        # Main agent loop
        while self.step_count < self.max_steps and not self.is_done:
            self.step_count += 1
            self._log(f"\n{'='*60}", prefix="")
            self._log(f"Step {self.step_count}/{self.max_steps}", prefix="📍")

            try:
                # Call LLM with tool definitions
                response = completion(
                    model=self.model_name,
                    messages=self.messages,
                    tools=self.tool_schema,
                    tool_choice="auto"
                )

                assistant_message = response.choices[0].message

                # Add assistant message to history
                self.messages.append(assistant_message.model_dump())

                # Check if assistant wants to call a tool
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    # Process tool calls
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        self._log(f"Calling tool: {tool_name}", prefix="🔧")
                        self._log(f"Arguments: {json.dumps(tool_args, indent=2)}", prefix="  ")

                        # Execute the tool
                        result = self._execute_tool(tool_name, tool_args)

                        self._log(f"Result: {json.dumps(result, indent=2)}", prefix="  ")

                        # Check if this is finish_task
                        if tool_name == "finish_task" and result.get("success"):
                            self.is_done = True
                            self._log("Task marked as complete!", prefix="✅")

                        # Add tool result to messages
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps(result)
                        })

                        # Break if done
                        if self.is_done:
                            break

                elif assistant_message.content:
                    # Assistant returned text (thinking/reasoning)
                    self._log(f"Agent thought: {assistant_message.content}", prefix="💭")

                else:
                    # No tool call and no content - something went wrong
                    self._log("Warning: No tool call or content in response", prefix="⚠️")
                    break

            except Exception as e:
                self._log(f"Error in agent loop: {str(e)}", prefix="❌")
                return {
                    "success": False,
                    "error": str(e),
                    "steps": self.step_count,
                    "is_done": self.is_done
                }

        # Final status
        if self.is_done:
            self._log(f"\n{'='*60}", prefix="")
            self._log(f"Task completed successfully in {self.step_count} steps", prefix="✅")
            return {
                "success": True,
                "steps": self.step_count,
                "is_done": True,
                "reason": "Agent called finish_task"
            }
        else:
            self._log(f"\n{'='*60}", prefix="")
            self._log(f"Task stopped after {self.step_count} steps (max reached)", prefix="⚠️")
            return {
                "success": False,
                "steps": self.step_count,
                "is_done": False,
                "reason": "Max steps reached"
            }

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.messages

    def reset(self) -> None:
        """Reset the agent state."""
        self.messages = []
        self.is_done = False
        self.step_count = 0
