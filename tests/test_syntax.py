"""
Simple smoke test without requiring dependencies to be installed.
Tests basic code structure and imports.
"""

import sys
import ast
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent


def test_file_syntax(filepath):
    """Test if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, "OK"
    except SyntaxError as e:
        return False, str(e)


def main():
    """Run syntax tests on all Python files."""
    print("=" * 60)
    print("Running syntax tests on all Python files")
    print("=" * 60)
    
    # Find all Python files
    python_files = list(BASE_DIR.rglob('*.py'))
    python_files = [f for f in python_files if 'venv' not in str(f) and '.eggs' not in str(f)]
    
    passed = 0
    failed = 0
    
    for filepath in sorted(python_files):
        relative_path = filepath.relative_to(BASE_DIR)
        success, message = test_file_syntax(filepath)
        
        if success:
            print(f"✓ {relative_path}")
            passed += 1
        else:
            print(f"✗ {relative_path}: {message}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} files")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All syntax tests passed!")


if __name__ == '__main__':
    main()
