"""Tool registry for registering and safely executing agent tools.

Design Principles:
1. Declarative Registration: Use `@registry.tool(...)` with Pydantic input models.
2. OpenAI Schema Export: `registry.schemas()` outputs the standard
   `[{"type": "function", "function": {"name", "description", "parameters"}}]` list.
3. Resilience (NEVER raises): `registry.run(name, args)` catches validation errors,
   runtime exceptions, and timeouts, returning `(result_str, is_error)`.
   This allows the LLM to observe its own error and self-correct on the next iteration.
"""

import inspect
import json
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, ValidationError


class RegisteredTool:
    """Represents a single registered tool and its metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        input_model: Type[BaseModel],
        func: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.input_model = input_model
        self.func = func

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool definition to OpenAI function tool call schema."""
        # Pydantic v2 model_json_schema() provides standard JSON schema
        schema = self.input_model.model_json_schema()
        # Clean title & descriptions if necessary for standard OpenAI format
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    """Manages tool registration, schema generation, and safe execution."""

    def __init__(self):
        self._tools: Dict[str, RegisteredTool] = {}

    def tool(
        self,
        name: str,
        description: str,
        input_model: Type[BaseModel],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a tool.
        
        Example:
            @registry.tool(
                name="flag_abnormal",
                description="Checks values against reference ranges",
                input_model=FlagAbnormalInput,
            )
            def flag_abnormal(rows: list[dict]) -> dict:
                ...
        """
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                raise ValueError(f"Tool '{name}' is already registered.")
            self._tools[name] = RegisteredTool(
                name=name,
                description=description.strip(),
                input_model=input_model,
                func=func,
            )
            return func

        return decorator

    def get_tool(self, name: str) -> Optional[RegisteredTool]:
        """Fetch registered tool by name."""
        return self._tools.get(name)

    def schemas(self) -> List[Dict[str, Any]]:
        """Return list of all registered tools in OpenAI function format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def run(self, name: str, args: Any) -> Tuple[str, bool]:
        """Safely executes a tool by name with arguments.
        
        NEVER raises an exception:
        - If the tool is not found -> returns is_error=True
        - If arguments are not a dict or cannot be parsed -> returns is_error=True
        - If Pydantic validation fails -> returns validation error details
        - If the function raises an exception -> catches and returns error string
        
        Returns:
            Tuple of (content_string, is_error_bool).
        """
        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(self._tools.keys())
            return (
                f"Error: Tool '{name}' is not registered. Available tools: {available}",
                True,
            )

        # Ensure args is a dictionary
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception as e:
                return (f"Error: Invalid JSON string for tool arguments: {e}", True)

        if not isinstance(args, dict):
            return (
                f"Error: Tool arguments must be a dictionary/object, received: {type(args).__name__}",
                True,
            )

        # Validate arguments using Pydantic input model
        try:
            validated = tool.input_model(**args)
        except ValidationError as val_err:
            return (
                f"Validation Error for tool '{name}': {val_err.json()}",
                True,
            )
        except Exception as e:
            return (
                f"Parameter parsing error for tool '{name}': {str(e)}",
                True,
            )

        # Execute the tool function
        try:
            # Check function signature
            sig = inspect.signature(tool.func)
            if len(sig.parameters) == 1 and next(iter(sig.parameters.values())).annotation == tool.input_model:
                raw_result = tool.func(validated)
            else:
                # Use getattr to preserve sub-model instances rather than converting them to raw dicts
                field_kwargs = {name: getattr(validated, name) for name in type(validated).model_fields}
                raw_result = tool.func(**field_kwargs)

            # Format the output into a string for the LLM
            if isinstance(raw_result, (dict, list)):
                content = json.dumps(raw_result, indent=2, ensure_ascii=False)
            else:
                content = str(raw_result)
            return (content, False)

        except Exception as exc:
            err_trace = traceback.format_exc(limit=2)
            return (
                f"Runtime error executing tool '{name}': {str(exc)}\n{err_trace}",
                True,
            )


# Default global tool registry for convenient single-instance use
default_registry = ToolRegistry()
