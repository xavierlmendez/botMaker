import os
import sys

import pytest


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    tests_roots = [
        os.path.join(os.path.dirname(__file__), "mathDomain"),
        os.path.join(os.path.dirname(__file__), "mlDomain"),
    ]
    return pytest.main(tests_roots)


if __name__ == "__main__":
    raise SystemExit(main())
