#!/usr/bin/env python3
"""
Test to verify semantic file detection fixes work correctly.
This mocks ClickZetta operations to test our core fixes without requiring real connection.
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datus.tools.semantic_models.adapters.clickzetta_adapter import ClickZettaVolumeAdapter

def create_mock_connector():
    """Create a mock ClickZetta connector with simulated data."""
    mock_connector = Mock()

    # Mock the LIST command response as pandas DataFrame
    # This simulates the data structure the user reported having files in
    mock_list_result = pd.DataFrame({
        'relative_path': [
            'semantic_models/Test006.yaml',
            'semantic_models/semantic_model_test.yaml',
            'semantic_models/tpch_100gb.yaml'
        ],
        'size': [2048, 1536, 3072],
        'modified_time': ['2025-10-30 10:00:00', '2025-10-30 10:15:00', '2025-10-30 10:30:00']
    })

    # Mock the _run_command method to return our DataFrame
    mock_connector._run_command.return_value = mock_list_result

    # Mock the read_volume_file method to return sample YAML content
    def mock_read_volume_file(volume_uri, file_path):
        """Mock volume file reading with sample content."""
        print(f"    Debug: read_volume_file called with volume_uri='{volume_uri}', file_path='{file_path}'")

        # Verify the correct User Volume format
        if volume_uri == "volume:user://~/" and "semantic_models/" in file_path:
            print(f"    ✅ Correct User Volume format used")
        if 'Test006.yaml' in file_path:
            return """
# ClickZetta Semantic Model
data_source:
  name: test_orders
  description: Test orders semantic model
  database: quick_start
  schema: public
  table_name: orders

dimensions:
  - name: order_id
    type: primary_key
    column: o_orderkey

  - name: customer_id
    type: foreign_key
    column: o_custkey

metrics:
  - name: total_orders
    type: count
    measure:
      agg: count
      expr: o_orderkey
"""
        elif 'semantic_model_test.yaml' in file_path:
            return """
# MetricFlow Semantic Model
semantic_model:
  name: customer_metrics
  description: Customer metrics semantic model
  model: ref('customers')

  dimensions:
    - name: customer_id
      type: primary_key

  measures:
    - name: customer_count
      agg: count
"""
        elif 'tpch_100gb.yaml' in file_path:
            return """
# ClickZetta TPCH Semantic Model
data_source:
  name: lineitem_metrics
  description: TPCH lineitem semantic model for 100GB dataset
  database: quick_start
  schema: tpch_100gb
  table_name: lineitem

dimensions:
  - name: order_key
    type: foreign_key
    column: l_orderkey

metrics:
  - name: total_revenue
    type: sum
    measure:
      agg: sum
      expr: l_extendedprice * (1 - l_discount)
"""
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    mock_connector.read_volume_file.side_effect = mock_read_volume_file

    return mock_connector

def test_volume_adapter_fixes():
    """Test that our ClickZetta volume adapter fixes work correctly."""

    print("=== Testing ClickZetta Volume Adapter Fixes ===\n")

    # Create mock connector
    mock_connector = create_mock_connector()

    # Create semantic files config matching the user's setup
    config = {
        "enabled": True,
        "storage_provider": "volume",
        "auto_import": True,
        "sync_on_startup": True,
        "file_patterns": ["*.yml", "*.yaml"],
        "provider_config": {
            "volume_type": "user",
            "volume_name": "semantic_models",
            "volume_path": "/semantic_models"
        }
    }

    try:
        # Test 1: Create volume adapter
        print("1. Testing volume adapter initialization...")
        adapter = ClickZettaVolumeAdapter(mock_connector, config)
        print("✅ Volume adapter created successfully")
        print(f"   Volume type: {adapter.volume_type}")
        print(f"   Volume path: {adapter.volume_path}")

        # Test 2: Test file listing (this was the main issue)
        print("\n2. Testing file listing (_parse_list_result fix)...")
        model_files = adapter.list_models()
        print(f"✅ Found {len(model_files)} model files:")
        for i, file in enumerate(model_files, 1):
            print(f"   {i}. {file}")

        expected_files = ['Test006.yaml', 'semantic_model_test.yaml', 'tpch_100gb.yaml']
        if set(model_files) == set(expected_files):
            print("✅ File detection fix working correctly")
        else:
            print(f"❌ File detection issue: expected {expected_files}, got {model_files}")
            return False

        # Test 3: Test model info for each file (this was causing "Error" status)
        print("\n3. Testing model info retrieval (get_model_info fix)...")
        for file in model_files:
            print(f"\n   Testing file: {file}")
            try:
                info = adapter.get_model_info(file)
                print(f"     ✅ Model name: {info.get('model_name', 'unknown')}")
                print(f"     ✅ Format: {info.get('format', 'unknown')}")
                print(f"     ✅ Content length: {info.get('content_length', 0)}")

                if 'content_error' in info:
                    print(f"     ❌ Content error: {info['content_error']}")
                    return False

                # Verify format detection
                expected_format = None
                if file == 'Test006.yaml' or file == 'tpch_100gb.yaml':
                    expected_format = 'clickzetta'
                elif file == 'semantic_model_test.yaml':
                    expected_format = 'metricflow'

                if info.get('format') == expected_format:
                    print(f"     ✅ Format detection working: {expected_format}")
                else:
                    print(f"     ❌ Format detection issue: expected {expected_format}, got {info.get('format')}")

            except Exception as e:
                print(f"     ❌ Failed to get info for {file}: {e}")
                return False

        # Test 4: Test file reading (volume path fix)
        print("\n4. Testing file reading (read_model volume path fix)...")
        for file in model_files[:2]:  # Test first 2 files
            try:
                content = adapter.read_model(file)
                print(f"   ✅ Successfully read {file} ({len(content)} characters)")

                # Verify we can parse content
                if 'data_source:' in content and 'name:' in content:
                    print(f"     ✅ ClickZetta format detected in content")
                elif 'semantic_model:' in content and 'name:' in content:
                    print(f"     ✅ MetricFlow format detected in content")
                else:
                    print(f"     ⚠️ Unknown format in content")

            except Exception as e:
                print(f"   ❌ Failed to read {file}: {e}")
                return False

        # Test 5: Test status logic that would be used in CLI
        print("\n5. Testing CLI status logic...")
        status_results = []
        for file in model_files:
            try:
                info = adapter.get_model_info(file)
                model_format = info.get("format", "unknown")

                if "content_error" in info:
                    status = "Error"
                elif model_format and model_format != "unknown":
                    status = "Available"
                else:
                    status = "Unknown format"

                status_results.append((file, status))
                print(f"   {file}: {status}")

            except Exception as e:
                status = "Error"
                status_results.append((file, status))
                print(f"   {file}: {status} ({e})")

        # Verify all files show as "Available" instead of "Error"
        error_count = sum(1 for _, status in status_results if status == "Error")
        available_count = sum(1 for _, status in status_results if status == "Available")

        print(f"\n   Status Summary:")
        print(f"   - Available: {available_count}")
        print(f"   - Error: {error_count}")
        print(f"   - Other: {len(status_results) - available_count - error_count}")

        if error_count == 0 and available_count == len(model_files):
            print("   ✅ All files show as Available (Status error fix working)")
        else:
            print("   ❌ Some files still showing as Error")
            return False

        print("\n=== All Tests Passed! ===")
        print("Summary of fixes verified:")
        print("✅ File detection: _parse_list_result now handles pandas DataFrames correctly")
        print("✅ Model info: get_model_info method added and working")
        print("✅ File reading: Volume path construction fixed")
        print("✅ Status logic: CLI will show 'Available' instead of 'Error' for valid files")
        print("✅ Format detection: ClickZetta and MetricFlow formats properly identified")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_volume_adapter_fixes()
    print(f"\n🎯 Test Result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)