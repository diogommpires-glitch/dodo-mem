"""Start the local broker, proxy and resumable organisation worker."""

import argparse
import asyncio

from .service import STATE, serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-index", action="store_true")
    args = parser.parse_args()
    asyncio.run(serve(STATE, index=not args.no_index))


if __name__ == "__main__":
    main()
