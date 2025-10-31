#!/usr/bin/env python3
"""
Debug the specific volume path issue by analyzing the error message.
"""

def analyze_volume_path_error():
    """Analyze the volume path error from the CLI output."""

    print("=== Volume Path Error Analysis ===\n")

    # The error from CLI output
    error_path = "volume:user/semantic_models/semantic_model_test.yaml"
    print(f"Error path: '{error_path}'")

    # Expected format based on _normalize_volume_uri
    print("\nAnalyzing _normalize_volume_uri logic:")
    print("- Input volume: 'volume:user'")
    print("- Input relative_path: 'semantic_models/semantic_model_test.yaml'")
    print("- Expected result: 'volume:user/semantic_models/semantic_model_test.yaml'")

    # The error shows exactly this format, so the issue might be elsewhere
    print("\n🤔 The path format looks correct according to _normalize_volume_uri...")

    # Let's check what might be wrong
    print("\nPossible issues:")
    print("1. The volume 'user' might not exist or not be accessible")
    print("2. The path 'semantic_models/semantic_model_test.yaml' might not exist")
    print("3. Permissions issue accessing the volume")
    print("4. ClickZetta session/authentication issue")

    # Check the difference between our test and real environment
    print("\nDifference from our test:")
    print("- Our test used: volume_uri='volume:user', file_path='semantic_models/...'")
    print("- Real error shows: 'volume:user/semantic_models/...' (combined)")
    print("- This suggests _normalize_volume_uri is working correctly")

    print("\n💡 Next steps:")
    print("1. Verify the actual LIST command output in real environment")
    print("2. Check if the files actually exist at the reported paths")
    print("3. Test volume access permissions")
    print("4. Add more detailed logging to the adapter")

if __name__ == "__main__":
    analyze_volume_path_error()