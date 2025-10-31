#!/usr/bin/env python3
"""
Debug script to understand what's happening in the real ClickZetta environment.
This will help us identify the exact issues causing the problems.
"""

import os
import sys
import yaml
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_clickzetta_integration():
    """Debug the actual ClickZetta integration step by step."""

    print("=== ClickZetta Semantic Integration Debug ===\n")

    try:
        # Step 1: Check environment variables
        print("1. Checking ClickZetta environment variables...")
        required_vars = [
            "CLICKZETTA_SERVICE", "CLICKZETTA_USERNAME", "CLICKZETTA_PASSWORD",
            "CLICKZETTA_INSTANCE", "CLICKZETTA_WORKSPACE", "CLICKZETTA_SCHEMA",
            "CLICKZETTA_VCLUSTER"
        ]

        env_status = {}
        for var in required_vars:
            value = os.getenv(var)
            env_status[var] = bool(value and not value.startswith('${'))
            print(f"   {var}: {'✓' if env_status[var] else '✗'}")

        if not all(env_status.values()):
            print("❌ Missing required environment variables")
            return False

        # Step 2: Load configuration
        print("\n2. Loading configuration...")
        config_path = "conf/agent.clickzetta.yml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        agent_config = config['agent']
        clickzetta_config = agent_config['namespace']['clickzetta']
        semantic_config = agent_config.get('external_semantic_files', {})

        print(f"   ✓ Config loaded")
        print(f"   External semantic files enabled: {semantic_config.get('enabled', False)}")
        print(f"   Volume type: {semantic_config.get('provider_config', {}).get('volume_type', 'unknown')}")
        print(f"   Volume path: {semantic_config.get('provider_config', {}).get('volume_path', 'unknown')}")

        # Step 3: Create ClickZetta connector
        print("\n3. Creating ClickZetta connector...")
        from datus.tools.db_tools.clickzetta_connector import ClickzettaConnector

        connector = ClickzettaConnector(
            service=os.getenv('CLICKZETTA_SERVICE'),
            username=os.getenv('CLICKZETTA_USERNAME'),
            password=os.getenv('CLICKZETTA_PASSWORD'),
            instance=os.getenv('CLICKZETTA_INSTANCE'),
            workspace=os.getenv('CLICKZETTA_WORKSPACE'),
            schema=os.getenv('CLICKZETTA_SCHEMA'),
            vcluster=os.getenv('CLICKZETTA_VCLUSTER'),
            secure=clickzetta_config.get('secure', False)
        )
        print("   ✓ ClickZetta connector created")

        # Step 4: Test semantic integration
        print("\n4. Testing semantic integration...")
        semantic_integration = connector.semantic_integration
        if semantic_integration:
            print("   ✓ Semantic integration available")
        else:
            print("   ❌ Semantic integration not available")
            return False

        # Step 5: Test volume adapter directly
        print("\n5. Testing volume adapter...")
        from datus.tools.semantic_models.adapters.clickzetta_adapter import ClickZettaVolumeAdapter

        adapter = ClickZettaVolumeAdapter(connector, semantic_config)
        print(f"   ✓ Volume adapter created")
        print(f"   Volume type: {adapter.volume_type}")
        print(f"   Volume name: {adapter.volume_name}")
        print(f"   Volume path: {adapter.volume_path}")

        # Step 6: Test LIST command directly
        print("\n6. Testing LIST command...")
        try:
            volume_prefix = adapter._get_volume_prefix()
            print(f"   Volume prefix: {volume_prefix}")

            if adapter.volume_path.strip("/"):
                sql = f"LIST {volume_prefix} SUBDIRECTORY '{adapter.volume_path.strip('/')}'"
            else:
                sql = f"LIST {volume_prefix}"

            print(f"   SQL command: {sql}")

            result = connector._run_command(sql)
            print(f"   ✓ LIST command executed")
            print(f"   Result type: {type(result)}")

            # Check if result has pandas methods
            if hasattr(result, 'to_pandas'):
                print("   Result has to_pandas method")
                df = result.to_pandas()
            elif hasattr(result, 'columns'):
                print("   Result is already a pandas DataFrame")
                df = result
            else:
                print(f"   ❌ Unknown result format: {type(result)}")
                return False

            print(f"   DataFrame shape: {df.shape}")
            print(f"   DataFrame columns: {list(df.columns)}")

            if len(df) > 0:
                print("   First few rows:")
                print(df.head())
            else:
                print("   ❌ No files found in LIST result")
                return False

        except Exception as e:
            print(f"   ❌ LIST command failed: {e}")
            traceback.print_exc()
            return False

        # Step 7: Test file parsing
        print("\n7. Testing file parsing...")
        try:
            files = adapter._parse_list_result(result)
            print(f"   Parsed files: {files}")

            if not files:
                print("   ❌ No files parsed from result")
                return False

        except Exception as e:
            print(f"   ❌ File parsing failed: {e}")
            traceback.print_exc()
            return False

        # Step 8: Test file reading
        print("\n8. Testing file reading...")
        if files:
            test_file = files[0]
            print(f"   Testing file: {test_file}")

            try:
                # Debug the exact parameters being passed
                volume_path = adapter.volume_path.strip("/")
                file_path = f"{volume_path}/{test_file}"
                volume_uri = "volume:user"

                print(f"   Volume URI: '{volume_uri}'")
                print(f"   File path: '{file_path}'")

                content = connector.read_volume_file(volume_uri, file_path)
                print(f"   ✓ File read successfully: {len(content)} characters")

                # Show first 200 characters
                print(f"   Content preview: {content[:200]}...")

            except Exception as e:
                print(f"   ❌ File reading failed: {e}")
                traceback.print_exc()
                return False

        # Step 9: Test get_model_info
        print("\n9. Testing get_model_info...")
        if files:
            test_file = files[0]
            try:
                info = adapter.get_model_info(test_file)
                print(f"   Model info: {info}")

                if info.get('format') == 'unknown':
                    print("   ❌ Format detection failed")
                    print(f"   Content error: {info.get('content_error', 'None')}")
                else:
                    print(f"   ✓ Format detected: {info.get('format')}")

            except Exception as e:
                print(f"   ❌ get_model_info failed: {e}")
                traceback.print_exc()
                return False

        print("\n✅ Debug completed successfully")
        return True

    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_clickzetta_integration()
    print(f"\n🎯 Debug Result: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)