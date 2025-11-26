#!/usr/bin/env python
import sys
sys.path.insert(0, '/Users/liangmo/Documents/GitHub/Datus-agent-clickzetta')

# Test if tools pattern detection works
class MockInput:
    tools = 'db_tools.*, context_search_tools.*, mcp_tool.*'

input_obj = MockInput()
tools_pattern = getattr(input_obj, 'tools', '') if input_obj else ''

print(f"Tools pattern: '{tools_pattern}'")
print(f"Contains mcp_tool: {'mcp_tool' in tools_pattern}")

if tools_pattern and 'mcp_tool' in tools_pattern:
    print("✅ Would load MCP tools!")
else:
    print("❌ Would NOT load MCP tools")
