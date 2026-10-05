"""Allow running as: python -m bookscan_sk"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
