"""Minimal HELENA resource-management command line."""

import argparse
import sys

from .runtime_cache import (
    BetheHeitlerCacheResourceUnavailable,
    BetheHeitlerResponseCacheInvalid,
    BetheHeitlerResponseCacheMiss,
    bethe_heitler_cache_directory,
    fetch_official_bethe_heitler_cache,
    validate_official_bethe_heitler_cache,
)


def _parser():
    parser = argparse.ArgumentParser(prog="helena")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="manage the validated BH cache")
    actions = cache.add_subparsers(dest="action", required=True)
    for name in ("fetch", "status", "path"):
        action = actions.add_parser(name)
        action.add_argument(
            "--cache-dir",
            help="custom BH cache directory for offline, HPC, or shared storage",
        )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    directory = bethe_heitler_cache_directory(args.cache_dir)
    if args.action == "path":
        print(directory)
        return 0
    if args.action == "status":
        try:
            validate_official_bethe_heitler_cache(directory)
        except BetheHeitlerResponseCacheMiss as error:
            print(f"missing: {error}")
            return 1
        except BetheHeitlerResponseCacheInvalid as error:
            print(f"invalid: {error}")
            return 1
        print(f"compatible: {directory}")
        return 0
    try:
        installed = fetch_official_bethe_heitler_cache(directory)
    except (
        BetheHeitlerCacheResourceUnavailable,
        BetheHeitlerResponseCacheInvalid,
        BetheHeitlerResponseCacheMiss,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"helena cache fetch: {error}", file=sys.stderr)
        return 1
    print(installed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
