"""AIvilization entry point."""
from __future__ import annotations

import asyncio
import sys


def main() -> None:
    """Main entry point for the aivilization CLI."""
    from aivilization.cli.app import AIvilizationCLI

    cli = AIvilizationCLI()
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
