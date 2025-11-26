#!/usr/bin/env python
"""Test script to verify MCP tools setup"""

import sys
import os

# Add the project to path
sys.path.insert(0, '/Users/liangmo/Documents/GitHub/Datus-agent-clickzetta')

print("=" * 80)
print("Testing MCP Tools Setup")
print("=" * 80)

# Load configuration
from datus.configuration.agent_config_loader import load_agent_config

config_path = 'conf/agent.universal-mcp-english.yml'
print(f"\n1. Loading config from: {config_path}")
agent_config = load_agent_config(config=config_path)

# Check agentic_nodes configuration
print("\n2. Checking agentic_nodes configuration...")
if hasattr(agent_config, 'agentic_nodes'):
    nodes = agent_config.agentic_nodes
    for node_name, node_config in nodes.items():
        print(f"\n   Node: {node_name}")
        if 'tools' in node_config:
            print(f"   Tools: {node_config['tools']}")
        else:
            print(f"   Tools: NOT FOUND")

# Try to create a ChatAgenticNode for clickzetta_mcp_server_220_sse
print("\n3. Creating ChatAgenticNode for 'clickzetta_mcp_server_220_sse'...")
from datus.agent.node.chat_agentic_node import ChatAgenticNode
from datus.schemas.chat_agentic_node_models import ChatNodeInput

# Get node config
node_config = agent_config.agentic_nodes.get('clickzetta_mcp_server_220_sse', {})
print(f"   Node config: {node_config}")

# Create input with tools spec
tools_spec = node_config.get('tools', '')
print(f"   Tools spec from config: '{tools_spec}'")

chat_input = ChatNodeInput(
    user_message="test message",
    tools=tools_spec
)
print(f"   Created ChatNodeInput with tools: '{chat_input.tools}'")

# Create node
try:
    node = ChatAgenticNode(
        node_id="test_node",
        description="Test node",
        node_type="chat",
        input_data=chat_input,
        agent_config=agent_config
    )

    print(f"\n4. Node created successfully!")
    print(f"   Total tools count: {len(node.tools)}")
    print(f"   MCP tools count: {len(node.mcp_tools)}")

    print("\n5. Tool breakdown:")
    db_tool_count = len(node.db_func_tool.available_tools()) if node.db_func_tool else 0
    context_tool_count = len(node.context_search_tools.available_tools()) if node.context_search_tools else 0
    print(f"   - DB tools: {db_tool_count}")
    print(f"   - Context tools: {context_tool_count}")
    print(f"   - MCP tools: {len(node.mcp_tools)}")

    if node.mcp_tools:
        print("\n6. MCP tools found:")
        for i, tool in enumerate(node.mcp_tools[:5], 1):  # Show first 5
            if hasattr(tool, 'name'):
                print(f"   {i}. {tool.name}")
            elif isinstance(tool, dict) and 'function' in tool:
                print(f"   {i}. {tool['function'].get('name', 'unnamed')}")
            else:
                print(f"   {i}. {type(tool)}")
    else:
        print("\n6. NO MCP TOOLS FOUND!")
        print("   This is the problem we need to fix.")

except Exception as e:
    import traceback
    print(f"\n ERROR: Failed to create node: {e}")
    print(traceback.format_exc())

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
