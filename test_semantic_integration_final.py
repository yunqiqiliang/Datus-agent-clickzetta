#!/usr/bin/env python3
"""
Final test script to verify semantic file integration fixes.
This tests the complete flow from file detection to status checking.
"""

import os
import sys
import yaml
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datus.tools.db_tools.clickzetta_connector import ClickzettaConnector
from datus.tools.semantic_models.adapters.clickzetta_adapter import ClickZettaVolumeAdapter
from datus.cli.semantic_file_commands import SemanticFileCommands

def test_semantic_integration():
    """Test the complete semantic file integration."""

    print("=== Testing Semantic File Integration ===\n")

    # Load configuration
    config_path = "conf/agent.clickzetta.yml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Test 1: Initialize ClickZetta connector
    print("1. Testing ClickZetta connector initialization...")
    try:
        clickzetta_config = config['namespace']['clickzetta']
        connector = ClickzettaConnector(clickzetta_config)
        print("✅ ClickZetta connector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ClickZetta connector: {e}")
        return False

    # Test 2: Check semantic integration availability
    print("\n2. Testing semantic integration availability...")
    try:
        integration = connector.semantic_integration
        if integration:
            print("✅ Semantic integration is available")
        else:
            print("❌ Semantic integration is not available")
            return False
    except Exception as e:
        print(f"❌ Error accessing semantic integration: {e}")
        return False

    # Test 3: Test volume adapter directly
    print("\n3. Testing ClickZetta volume adapter...")
    try:
        semantic_config = config.get('external_semantic_files', {})
        adapter = ClickZettaVolumeAdapter(connector, semantic_config)
        print("✅ Volume adapter initialized successfully")

        # Test volume configuration
        print(f"   Volume type: {adapter.volume_type}")
        print(f"   Volume name: {adapter.volume_name}")
        print(f"   Volume path: {adapter.volume_path}")

    except Exception as e:
        print(f"❌ Failed to initialize volume adapter: {e}")
        return False

    # Test 4: List models
    print("\n4. Testing file listing...")
    try:
        model_files = adapter.list_models()
        print(f"✅ Found {len(model_files)} model files:")
        for i, file in enumerate(model_files, 1):
            print(f"   {i}. {file}")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return False

    # Test 5: Test model info for each file
    print("\n5. Testing model info for each file...")
    for file in model_files[:3]:  # Test first 3 files
        try:
            print(f"\n   Testing file: {file}")
            info = adapter.get_model_info(file)

            print(f"     Model name: {info.get('model_name', 'unknown')}")
            print(f"     Format: {info.get('format', 'unknown')}")
            print(f"     Content length: {info.get('content_length', 0)}")

            if 'content_error' in info:
                print(f"     ❌ Content error: {info['content_error']}")
            else:
                print(f"     ✅ File info retrieved successfully")

        except Exception as e:
            print(f"     ❌ Failed to get info for {file}: {e}")

    # Test 6: Test file reading
    print("\n6. Testing file reading...")
    if model_files:
        test_file = model_files[0]
        try:
            content = adapter.read_model(test_file)
            print(f"✅ Successfully read {test_file} ({len(content)} characters)")

            # Try to parse as YAML to verify format
            try:
                docs = list(yaml.safe_load_all(content))
                for doc in docs:
                    if isinstance(doc, dict):
                        if 'data_source' in doc:
                            print(f"   Format: ClickZetta semantic model")
                            print(f"   Model name: {doc.get('data_source', {}).get('name', 'unknown')}")
                            break
                        elif 'semantic_model' in doc:
                            print(f"   Format: MetricFlow semantic model")
                            print(f"   Model name: {doc.get('semantic_model', {}).get('name', 'unknown')}")
                            break
            except yaml.YAMLError as e:
                print(f"   ⚠️ YAML parsing error: {e}")

        except Exception as e:
            print(f"❌ Failed to read {test_file}: {e}")

    # Test 7: Test integration methods
    print("\n7. Testing integration methods...")
    try:
        available_models = integration.list_available_models()
        print(f"✅ Integration list_available_models: {len(available_models)} files")

        if available_models:
            test_file = available_models[0]
            model_info = integration.get_model_info(test_file)
            print(f"✅ Integration get_model_info for {test_file}:")
            print(f"   Format: {model_info.get('format', 'unknown')}")
            print(f"   Model name: {model_info.get('model_name', 'unknown')}")

    except Exception as e:
        print(f"❌ Failed to test integration methods: {e}")

    print("\n=== Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_semantic_integration()
    sys.exit(0 if success else 1)