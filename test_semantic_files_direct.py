#!/usr/bin/env python3
"""
Direct test for semantic files functionality.
This simulates the .semantic_files CLI command to test our fixes.
"""

import os
import sys
import yaml
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datus.tools.db_tools.clickzetta_connector import ClickzettaConnector
from datus.cli.semantic_file_commands import SemanticFileCommands

class MockAgentConfig:
    """Mock agent config."""
    def __init__(self, config_dict):
        self.config = config_dict
        self.current_namespace = "clickzetta"

        # Get the agent section which contains the actual config
        agent_config = config_dict.get("agent", {})

        # The config has "namespace" not "namespaces"
        self.namespace = agent_config.get("namespace", {})
        self.external_semantic_files_config = agent_config.get("external_semantic_files", {})

        # Debug print to see the config structure
        print(f"Debug: Config keys: {list(config_dict.keys())}")
        print(f"Debug: Agent config keys: {list(agent_config.keys())}")
        print(f"Debug: Namespace config: {self.namespace}")
        print(f"Debug: External semantic files config: {self.external_semantic_files_config}")

class MockCLI:
    """Mock CLI object for testing."""
    def __init__(self, agent_config):
        self.agent_config = agent_config
        # Simplified DB manager simulation
        self.db_manager = MockDBManager(agent_config)

class MockDBManager:
    """Mock DB manager for testing."""
    def __init__(self, agent_config):
        self.agent_config = agent_config
        self._connector = None

    def get_conn(self, namespace):
        """Get connection for namespace."""
        if not self._connector:
            namespace_config = self.agent_config.namespace[namespace]

            # Extract and substitute environment variables
            import os
            service = os.getenv('CLICKZETTA_SERVICE', namespace_config.get('service', ''))
            username = os.getenv('CLICKZETTA_USERNAME', namespace_config.get('username', ''))
            password = os.getenv('CLICKZETTA_PASSWORD', namespace_config.get('password', ''))
            instance = os.getenv('CLICKZETTA_INSTANCE', namespace_config.get('instance', ''))
            workspace = os.getenv('CLICKZETTA_WORKSPACE', namespace_config.get('workspace', ''))
            schema = os.getenv('CLICKZETTA_SCHEMA', namespace_config.get('schema', ''))
            vcluster = os.getenv('CLICKZETTA_VCLUSTER', namespace_config.get('vcluster', ''))
            secure = namespace_config.get('secure', False)

            # Remove ${...} variable substitution if environment variables are not set
            if service.startswith('${') and service.endswith('}'):
                service = ""
            if username.startswith('${') and username.endswith('}'):
                username = ""
            if password.startswith('${') and password.endswith('}'):
                password = ""
            if instance.startswith('${') and instance.endswith('}'):
                instance = ""
            if workspace.startswith('${') and workspace.endswith('}'):
                workspace = ""
            if schema.startswith('${') and schema.endswith('}'):
                schema = ""
            if vcluster.startswith('${') and vcluster.endswith('}'):
                vcluster = ""

            print(f"Debug: Creating ClickZetta connector with:")
            print(f"  service: {service}")
            print(f"  username: {username}")
            print(f"  instance: {instance}")
            print(f"  workspace: {workspace}")
            print(f"  schema: {schema}")
            print(f"  vcluster: {vcluster}")
            print(f"  secure: {secure}")

            if not all([service, username, password, instance, workspace]):
                raise ValueError(f"Missing required ClickZetta parameters: service={bool(service)}, username={bool(username)}, password={bool(password)}, instance={bool(instance)}, workspace={bool(workspace)}")

            self._connector = ClickzettaConnector(
                service=service,
                username=username,
                password=password,
                instance=instance,
                workspace=workspace,
                schema=schema,
                vcluster=vcluster,
                secure=secure
            )
        return self._connector

def test_semantic_files_command():
    """Test the semantic files command directly."""

    print("=== Testing Semantic Files Command ===\n")

    # Load configuration
    config_path = "conf/agent.clickzetta.yml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        # Load config manually
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create mock agent config
        agent_config = MockAgentConfig(config_dict)
        print("✅ Agent config loaded successfully")

        # Create mock CLI
        mock_cli = MockCLI(agent_config)
        print("✅ Mock CLI created successfully")

        # Create semantic file commands handler
        semantic_commands = SemanticFileCommands(mock_cli)
        print("✅ Semantic commands handler created successfully")

        # Test the list command (equivalent to .semantic_files)
        print("\n--- Testing .semantic_files command ---")
        semantic_commands.cmd_list_semantic_files()

        print("\n--- Testing .semantic_file_config command ---")
        semantic_commands.cmd_semantic_file_config()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_semantic_files_command()
    sys.exit(0 if success else 1)