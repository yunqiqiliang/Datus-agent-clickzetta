"""
LLM-Native MCP Handler

Based on OpenAI Codex MCP implementation - exposes MCP tools directly as functions to LLM
without semantic analysis or hardcoded mapping strategies. Follows Codex pattern:
1. Discover MCP tools from servers via MCPManager
2. Sanitize schemas for LLM consumption
3. Expose as native function calls
4. Route calls directly to MCP protocol via MCPManager

This provides a clean LLM-native interface while delegating all protocol handling
to the robust MCPManager implementation.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolCallStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class McpToolInfo:
    """MCP Tool Information following Codex pattern"""

    server_name: str
    tool_name: str
    qualified_name: str  # mcp__server__tool format
    description: str
    schema: Dict[str, Any]
    sanitized_schema: Dict[str, Any]


@dataclass
class ToolCallResult:
    """Tool execution result"""

    status: ToolCallStatus
    result: Any = None
    error: str = None
    duration: float = 0.0


class LLMNativeMCPHandler:
    """
    LLM-Native MCP Handler following OpenAI Codex architecture

    Key principles from Codex:
    1. No semantic analysis - let LLM choose tools directly
    2. Tool qualification: mcp__server__tool to avoid collisions
    3. Schema sanitization for LLM consumption
    4. Direct protocol execution without application-layer routing
    """

    def __init__(self, agent_config):
        self.agent_config = agent_config
        self.mcp_servers = {}  # server_name -> server_config
        self.qualified_tools = {}  # qualified_name -> McpToolInfo
        self.tool_lookup = {}  # qualified_name -> (server_name, tool_name)

        # Load MCP servers from config
        if hasattr(agent_config, "mcp_servers"):
            self.mcp_servers = agent_config.mcp_servers

    async def initialize(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Initialize MCP tools following Codex pattern:
        1. Connect to all MCP servers
        2. Discover tools from each server
        3. Qualify tool names to avoid collisions
        4. Sanitize schemas for LLM consumption
        5. Return OpenAI-compatible function definitions
        """
        logger.info("Initializing LLM-Native MCP Handler...")

        all_functions = {}

        for server_name, server_config in self.mcp_servers.items():
            try:
                logger.info(f"Discovering tools from MCP server: {server_name}")
                tools = await self._discover_server_tools(server_name, server_config)
                server_functions = []
                for tool in tools:
                    # Qualify tool name following Codex pattern
                    qualified_name = self._qualify_tool_name(server_name, tool["name"])

                    # Sanitize schema for LLM consumption
                    sanitized_schema = self._sanitize_tool_schema(tool.get("inputSchema", {}))

                    # Store tool info
                    tool_info = McpToolInfo(
                        server_name=server_name,
                        tool_name=tool["name"],
                        qualified_name=qualified_name,
                        description=tool.get("description", ""),
                        schema=tool.get("inputSchema", {}),
                        sanitized_schema=sanitized_schema,
                    )

                    self.qualified_tools[qualified_name] = tool_info
                    self.tool_lookup[qualified_name] = (server_name, tool["name"])

                    # Create OpenAI function definition
                    function_def = {
                        "name": qualified_name,
                        "description": tool.get("description", f"Execute {tool['name']} on {server_name}"),
                        "parameters": sanitized_schema,
                    }

                    server_functions.append(function_def)

                all_functions[server_name] = server_functions
                logger.info(f"Registered {len(server_functions)} tools from {server_name}")

            except Exception as e:
                logger.error(f"Failed to discover tools from {server_name}: {e}")
                all_functions[server_name] = []

        total_tools = sum(len(funcs) for funcs in all_functions.values())
        logger.info(f"LLM-Native MCP Handler initialized with {total_tools} total tools")

        return all_functions

    async def _discover_server_tools(self, server_name: str, server_config: Dict) -> List[Dict]:
        """
        Discover tools from MCP server using MCPManager

        MCPManager handles all protocol types correctly:
        - stdio: Process-based communication via stdin/stdout
        - sse: Server-Sent Events communication
        - http: HTTP-based communication

        No HTTP fallback needed - MCPManager is the authoritative source
        """
        try:
            logger.info(f"Discovering tools from MCP server {server_name} using MCPManager...")

            from datus.tools.mcp_tools.mcp_manager import MCPManager

            # Use MCPManager for all protocol handling
            mcp_manager = MCPManager()
            tools_result = mcp_manager.list_tools(server_name)

            if tools_result.success:
                tools = tools_result.result.get("tools", [])
                logger.info(f"MCPManager discovered {len(tools)} tools from {server_name}")

                # Validate and normalize tools
                normalized_tools = []
                for tool in tools:
                    if self._is_valid_tool(tool):
                        normalized_tools.append(self._validate_tool_schema(tool))
                    else:
                        logger.warning(f"Invalid tool definition from {server_name}: {tool}")

                return normalized_tools
            else:
                logger.warning(f"MCPManager failed to discover tools from {server_name}: {tools_result.message}")
                return []

        except ImportError:
            logger.error("MCPManager not available - this is required for MCP protocol handling")
            return []
        except Exception as e:
            logger.warning(f"Tool discovery failed for {server_name}: {e}")
            return []

    def _is_valid_tool(self, tool: Any) -> bool:
        """Validate that tool has required fields"""
        return isinstance(tool, dict) and "name" in tool and isinstance(tool["name"], str) and tool["name"].strip()

    def _validate_tool_schema(self, tool: Dict) -> Dict:
        """Ensure tool has all required fields with defaults"""
        return {
            "name": tool["name"],
            "description": tool.get("description", f"Execute {tool['name']}"),
            "inputSchema": tool.get("inputSchema", tool.get("input_schema", {})),
        }

    def _qualify_tool_name(self, server_name: str, tool_name: str) -> str:
        """
        Qualify tool name following Codex pattern: mcp__server__tool
        Ensures no collisions between tools from different servers
        """
        qualified = f"mcp__{server_name}__{tool_name}"

        # Handle length limits like Codex (max 64 chars with SHA1 suffix if needed)
        if len(qualified) > 64:
            # Use hash suffix for very long names
            hash_suffix = hashlib.sha1(qualified.encode()).hexdigest()[:8]
            max_prefix_len = 64 - 1 - 8  # 1 for underscore, 8 for hash
            qualified = f"{qualified[:max_prefix_len]}_{hash_suffix}"

        return qualified

    def _sanitize_tool_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize tool schema for LLM consumption following Codex pattern
        Fixes common MCP server schema quirks:
        1. Missing type fields
        2. Invalid type unions
        3. Missing properties for objects
        """
        if not schema:
            return {"type": "object", "properties": {}}

        def sanitize_recursive(obj: Any) -> Any:
            if not isinstance(obj, dict):
                return obj

            result = obj.copy()

            # Fix missing type field - infer from keywords
            if "type" not in result:
                if "properties" in result:
                    result["type"] = "object"
                elif "items" in result:
                    result["type"] = "array"
                elif "enum" in result:
                    result["type"] = "string"
                else:
                    result["type"] = "string"  # Default fallback

            # Handle type unions - pick first valid type
            if isinstance(result.get("type"), list):
                result["type"] = result["type"][0]

            # Ensure objects have properties
            if result.get("type") == "object" and "properties" not in result:
                result["properties"] = {}

            # Recursively sanitize nested schemas
            if "properties" in result:
                result["properties"] = {k: sanitize_recursive(v) for k, v in result["properties"].items()}

            if "items" in result:
                result["items"] = sanitize_recursive(result["items"])

            return result

        return sanitize_recursive(schema)

    async def execute_tool_call(self, qualified_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """
        Execute MCP tool call following Codex pattern
        Direct protocol execution without semantic analysis
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Resolve qualified name to server and tool
            if qualified_name not in self.tool_lookup:
                return ToolCallResult(
                    status=ToolCallStatus.ERROR,
                    error=f"Tool {qualified_name} not found",
                    duration=asyncio.get_event_loop().time() - start_time,
                )

            server_name, tool_name = self.tool_lookup[qualified_name]
            logger.info(f"Executing MCP tool: {server_name}.{tool_name} with args: {arguments}")

            # Get server config
            server_config = self.mcp_servers.get(server_name)
            if not server_config:
                return ToolCallResult(
                    status=ToolCallStatus.ERROR,
                    error=f"Server {server_name} not configured",
                    duration=asyncio.get_event_loop().time() - start_time,
                )

            # Execute tool via MCP protocol
            result = await self._execute_mcp_tool(server_config, tool_name, arguments)

            return ToolCallResult(
                status=ToolCallStatus.SUCCESS, result=result, duration=asyncio.get_event_loop().time() - start_time
            )

        except asyncio.TimeoutError:
            return ToolCallResult(
                status=ToolCallStatus.TIMEOUT,
                error="Tool execution timeout",
                duration=asyncio.get_event_loop().time() - start_time,
            )
        except Exception as e:
            logger.exception(f"Tool execution failed: {e}")
            return ToolCallResult(
                status=ToolCallStatus.ERROR, error=str(e), duration=asyncio.get_event_loop().time() - start_time
            )

    async def _execute_mcp_tool(self, server_config: Dict, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute tool via MCP protocol using MCPManager

        MCPManager handles all protocol types correctly and is the authoritative source
        for MCP tool execution. No HTTP fallback needed.
        """
        # Extract server name for MCPManager lookup
        server_name = None
        for name, config in self.mcp_servers.items():
            if config == server_config:
                server_name = name
                break

        if not server_name:
            # Find server name by URL if exact config match fails
            server_url = server_config.get("url", "")
            for name, config in self.mcp_servers.items():
                if config.get("url") == server_url:
                    server_name = name
                    break

        if not server_name:
            raise Exception("Cannot identify server name for MCP tool execution")

        try:
            from datus.tools.mcp_tools.mcp_manager import MCPManager

            mcp_manager = MCPManager()
            result = mcp_manager.call_tool(server_name, tool_name, arguments)

            if result.success:
                logger.info(f"MCPManager executed {tool_name} on {server_name} successfully")
                return result.result
            else:
                raise Exception(f"MCP tool execution failed: {result.message}")

        except ImportError:
            raise Exception("MCPManager not available - this is required for MCP protocol handling")
        except Exception as e:
            if "MCP tool execution failed:" in str(e):
                raise  # Re-raise MCPManager errors as-is
            else:
                raise Exception(f"MCP tool execution error: {e}")

    def get_all_function_definitions(self) -> List[Dict[str, Any]]:
        """Get all function definitions for LLM consumption"""
        functions = []
        for tool_info in self.qualified_tools.values():
            function_def = {
                "name": tool_info.qualified_name,
                "description": tool_info.description,
                "parameters": tool_info.sanitized_schema,
            }
            functions.append(function_def)
        return functions

    def get_tool_info(self, qualified_name: str) -> Optional[McpToolInfo]:
        """Get detailed tool information"""
        return self.qualified_tools.get(qualified_name)

    def list_available_tools(self) -> Dict[str, List[str]]:
        """List all available tools grouped by server"""
        tools_by_server = {}
        for tool_info in self.qualified_tools.values():
            if tool_info.server_name not in tools_by_server:
                tools_by_server[tool_info.server_name] = []
            tools_by_server[tool_info.server_name].append(tool_info.qualified_name)
        return tools_by_server
