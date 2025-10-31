#!/usr/bin/env python3
"""
Quick verification script to test specific semantic file fixes.
This script tests the core fixes without requiring full ClickZetta connection.
"""

import sys
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datus.tools.semantic_models.adapters.clickzetta_adapter import ClickZettaVolumeAdapter

def test_parse_list_result_fix():
    """Test the _parse_list_result fix for pandas DataFrame handling."""
    print("=== Testing _parse_list_result Fix ===")

    # Create a mock adapter
    from unittest.mock import Mock
    mock_connector = Mock()
    config = {
        "provider_config": {
            "volume_type": "user",
            "volume_name": "semantic_models",
            "volume_path": "/semantic_models"
        },
        "file_patterns": ["*.yml", "*.yaml"]
    }

    adapter = ClickZettaVolumeAdapter(mock_connector, config)

    # Test with pandas DataFrame (the actual format from ClickZetta)
    test_df = pd.DataFrame({
        'relative_path': [
            'semantic_models/Test006.yaml',
            'semantic_models/semantic_model_test.yaml',
            'semantic_models/tpch_100gb.yaml',
            'semantic_models/some_directory/',  # Should be filtered out
            'semantic_models/config.txt'        # Should be filtered out by pattern
        ]
    })

    print("Input DataFrame:")
    print(test_df)

    # Mock the _run_command to return our test DataFrame
    mock_connector._run_command.return_value = test_df

    # Test the complete list_models method (which includes filtering)
    try:
        result = adapter.list_models()
        print(f"\nFinal filtered result: {result}")

        expected = ['Test006.yaml', 'semantic_model_test.yaml', 'tpch_100gb.yaml']
        if set(result) == set(expected):
            print("✅ File listing and filtering working correctly")
            return True
        else:
            print(f"❌ Expected {expected}, got {result}")
            return False
    except Exception as e:
        print(f"❌ Error during list_models: {e}")
        return False

def test_volume_path_construction():
    """Test the volume path construction fix."""
    print("\n=== Testing Volume Path Construction Fix ===")

    from unittest.mock import Mock

    mock_connector = Mock()
    config = {
        "provider_config": {
            "volume_type": "user",
            "volume_name": "semantic_models",
            "volume_path": "/semantic_models"
        }
    }

    adapter = ClickZettaVolumeAdapter(mock_connector, config)

    # Track the calls to read_volume_file
    call_log = []

    def mock_read_volume_file(volume_uri, file_path):
        call_log.append((volume_uri, file_path))
        return "mock content"

    mock_connector.read_volume_file.side_effect = mock_read_volume_file

    # Test reading a file
    try:
        adapter.read_model("Test006.yaml")

        if call_log:
            volume_uri, file_path = call_log[0]
            print(f"Volume URI: '{volume_uri}'")
            print(f"File path: '{file_path}'")

            # Check if the fix is applied correctly
            if volume_uri == "volume:user://~/" and file_path == "semantic_models/Test006.yaml":
                print("✅ Volume path construction fix working correctly")
                print("   No more 'Invalid volume path' error should occur")
                return True
            else:
                print(f"❌ Incorrect path construction")
                print(f"   Expected: volume_uri='volume:user://~/', file_path='semantic_models/Test006.yaml'")
                print(f"   Got: volume_uri='{volume_uri}', file_path='{file_path}'")
                return False
        else:
            print("❌ read_volume_file was not called")
            return False

    except Exception as e:
        print(f"❌ Error during read_model: {e}")
        return False

def test_format_detection():
    """Test the format detection in get_model_info."""
    print("\n=== Testing Format Detection Fix ===")

    from unittest.mock import Mock

    mock_connector = Mock()
    config = {
        "provider_config": {
            "volume_type": "user",
            "volume_name": "semantic_models",
            "volume_path": "/semantic_models"
        }
    }

    adapter = ClickZettaVolumeAdapter(mock_connector, config)

    # Mock different content types
    test_cases = [
        ("Test006.yaml", """
data_source:
  name: test_orders
  description: Test orders semantic model
""", "clickzetta"),
        ("semantic_model_test.yaml", """
semantic_model:
  name: customer_metrics
  description: Customer metrics
""", "metricflow"),
        ("invalid.yaml", "invalid: yaml: content:", "unknown")
    ]

    def mock_read_model(file_name):
        for test_file, content, _ in test_cases:
            if test_file == file_name:
                return content
        raise FileNotFoundError(f"File not found: {file_name}")

    adapter.read_model = mock_read_model

    all_passed = True
    for file_name, content, expected_format in test_cases:
        try:
            info = adapter.get_model_info(file_name)
            actual_format = info.get('format', 'unknown')

            print(f"File: {file_name}")
            print(f"  Expected format: {expected_format}")
            print(f"  Actual format: {actual_format}")

            if actual_format == expected_format:
                print(f"  ✅ Format detection working")
            else:
                print(f"  ❌ Format detection failed")
                all_passed = False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_passed = False

    return all_passed

def main():
    """Run all verification tests."""
    print("Semantic File Integration Fixes - Quick Verification\n")

    results = []

    # Test 1: _parse_list_result fix
    results.append(test_parse_list_result_fix())

    # Test 2: Volume path construction fix
    results.append(test_volume_path_construction())

    # Test 3: Format detection fix
    results.append(test_format_detection())

    # Summary
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ All fixes verified successfully!")
        print("\nYour semantic file integration should now work correctly:")
        print("- Files will be detected from ClickZetta volume")
        print("- Status will show 'Available' instead of 'Error'")
        print("- File import should work without 'Invalid volume path' error")
    else:
        print("❌ Some fixes need attention")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)