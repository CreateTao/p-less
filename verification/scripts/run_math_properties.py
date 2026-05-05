"""Entry point: Step 1 - Run math property verification tests."""

import sys
import subprocess


def main():
    print("=== Step 1: Math Property Verification ===")
    print("Running pytest on verification/tests/ ...")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "verification/tests/", "-v", "--tb=short"],
        cwd=".",
    )

    if result.returncode == 0:
        print("\n✅ All math property tests PASSED")
    else:
        print("\n❌ Some math property tests FAILED - check output above")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())