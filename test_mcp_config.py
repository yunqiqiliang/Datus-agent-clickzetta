#!/usr/bin/env python
"""Test if agent_config has mcp_servers"""

import sys
sys.path.insert(0, '/Users/liangmo/Documents/GitHub/Datus-agent-clickzetta')

from datus.configuration.agent_config_loader import load_agent_config

config_path = 'conf/agent.universal-mcp-english.yml'
print(f"Loading config from: {config_path}")
agent_config = load_agent_config(config=config_path)

print(f"\nChecking agent_config attributes:")
print(f"  has 'mcp_servers': {hasattr(agent_config, 'mcp_servers')}")

if hasattr(agent_config, 'mcp_servers'):
    print(f"  mcp_servers type: {type(agent_config.mcp_servers)}")
    print(f"  mcp_servers value: {agent_config.mcp_servers}")
else:
    print("  ❌ agent_config does NOT have mcp_servers attribute!")
    print(f"\n  Available attributes: {[attr for attr in dir(agent_config) if not attr.startswith('_')]}")
