"""Minimal HELENA resource-management command line."""

import argparse
import sys

from .runtime_cache import (
    BetheHeitlerCacheResourceUnavailable,
    BetheHeitlerResponseCacheInvalid,
    BetheHeitlerResponseCacheMiss,
    bethe_heitler_cache_directory,
    fetch_current_c513_response_cache,
    fetch_official_bethe_heitler_cache,
    prepare_current_c513_response_cache,
    validate_official_bethe_heitler_cache,
)


def _parser():
    parser = argparse.ArgumentParser(prog="helena")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser(
        "cache", help="manage the validated BH and gamma-gamma caches"
    )
    actions = cache.add_subparsers(dest="action", required=True)
    for name in ("build", "fetch", "status", "path"):
        action = actions.add_parser(name)
        action.add_argument(
            "--cache-dir",
            help="custom response cache directory for offline, HPC, or shared storage",
        )
        if name in ("fetch", "status"):
            action.add_argument(
                "--legacy",
                action="store_true",
                help="use the immutable v0.1 resource; incompatible with current C513",
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
            if args.legacy:
                validate_official_bethe_heitler_cache(directory)
            else:
                prepare_current_c513_response_cache(directory)
        except BetheHeitlerResponseCacheMiss as error:
            print(f"missing: {error}")
            return 1
        except BetheHeitlerResponseCacheInvalid as error:
            print(f"invalid: {error}")
            return 1
        scope = "legacy v0.1 resource" if args.legacy else "current C513 grids"
        print(f"compatible: {scope}: {directory}")
        return 0
    try:
        if args.action == "build":
            print(
                "Preparing current C513 BH and gamma-gamma caches; a first build can take tens of minutes.",
                flush=True,
            )
            installed = prepare_current_c513_response_cache(directory, allow_build=True)
        elif args.legacy:
            installed = fetch_official_bethe_heitler_cache(directory)
        else:
            installed = fetch_current_c513_response_cache(directory)
    except (
        BetheHeitlerCacheResourceUnavailable,
        BetheHeitlerResponseCacheInvalid,
        BetheHeitlerResponseCacheMiss,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"helena cache {args.action}: {error}", file=sys.stderr)
        return 1
    print(installed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
