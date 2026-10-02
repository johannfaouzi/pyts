#!/usr/bin/env python3
"""Check that every UCR/UEA dataset can be downloaded and loaded by pyts.

This script calls pyts's own public API (``fetch_ucr_dataset`` and
``fetch_uea_dataset``) for every dataset name returned by
``ucr_dataset_list``/``uea_dataset_list`` (or a subset selected on the
command line), and reports which ones succeeded and which ones failed.

It is meant to be run locally, since it needs real network access to
``timeseriesclassification.com``. Downloading the full UCR + UEA archives
is a few GB and can take a long time, so by default datasets are
downloaded into a temporary directory that is deleted at the end of the
run; pass ``--keep`` to keep the downloaded files (e.g. to reuse the
default pyts cache, see ``--data-home``).

Examples
--------
Quick smoke test on a handful of small datasets from both archives::

    python scripts/check_dataset_downloads.py --limit 5

Full run of every UCR and UEA dataset, keeping the downloaded files in
pyts's default cache so they don't need to be re-downloaded later::

    python scripts/check_dataset_downloads.py --keep --use-default-cache

Only the UEA archive, with a 2 second delay between datasets and a report
written to a specific file::

    python scripts/check_dataset_downloads.py --archive uea --delay 2 \\
        --output uea_report.csv

"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import argparse
import csv
import shutil
import socket
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Try downloading and loading every UCR/UEA dataset with "
            "pyts's fetch_ucr_dataset/fetch_uea_dataset functions, and "
            "report which ones succeed or fail."
        )
    )
    parser.add_argument(
        '--archive',
        choices=('ucr', 'uea', 'both'),
        default='both',
        help="Which archive(s) to check (default: both).",
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        metavar='NAME',
        help=(
            "Only check these specific dataset names, instead of the "
            "full list. Overrides --limit."
        ),
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        metavar='N',
        help=(
            "Only check the first N datasets of each archive (useful for "
            "a quick smoke test). Default: check all of them."
        ),
    )
    parser.add_argument(
        '--data-home',
        default=None,
        metavar='PATH',
        help=(
            "Directory to download datasets into. Default: a fresh "
            "temporary directory, removed at the end of the run unless "
            "--keep is also given. See also --use-default-cache."
        ),
    )
    parser.add_argument(
        '--use-default-cache',
        action='store_true',
        help=(
            "Download into pyts's own default cache (the OS-appropriate "
            "per-user cache directory, e.g. ~/.cache/pyts/{UCR,UEA} on "
            "Linux), the same location used by "
            "fetch_ucr_dataset/fetch_uea_dataset when no data_home is "
            "given. Implies --keep. Ignored if --data-home is also given."
        ),
    )
    parser.add_argument(
        '--keep',
        action='store_true',
        help="Do not delete the downloaded files after the run.",
    )
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help=(
            "Reuse an already-downloaded copy of a dataset instead of "
            "downloading it again (fetch_*_dataset's use_cache=True). "
            "Default is to force a fresh download of every dataset, "
            "since the point of this script is to test the download "
            "itself. Useful to resume a previous --keep run."
        ),
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        metavar='SECONDS',
        help=(
            "Delay between two dataset downloads, to avoid hammering "
            "the server (default: 0.5)."
        ),
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        metavar='SECONDS',
        help="Per-request network timeout in seconds (default: 60).",
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=2,
        metavar='N',
        help=(
            "Number of extra attempts for a dataset whose download fails "
            "(default: 2, i.e. up to 3 attempts total). Set to 0 to "
            "disable retrying."
        ),
    )
    parser.add_argument(
        '--retry-delay',
        type=float,
        default=5.0,
        metavar='SECONDS',
        help=(
            "Delay before retrying a failed download (default: 5); "
            "doubles after each retry."
        ),
    )
    parser.add_argument(
        '--output',
        default=None,
        metavar='PATH',
        help=(
            "CSV report path. Default: "
            "dataset_download_report_<timestamp>.csv in the current "
            "directory."
        ),
    )
    return parser.parse_args(argv)


def _iter_targets(args):
    """Yield (archive, dataset_name) pairs to check."""
    from pyts.datasets import ucr_dataset_list, uea_dataset_list

    if args.datasets is not None:
        # The user gave explicit names: try them against whichever
        # archive(s) were requested, without filtering against the
        # official list (in case it is stale).
        archives = (
            ['ucr', 'uea'] if args.archive == 'both' else [args.archive]
        )
        for archive in archives:
            for name in args.datasets:
                yield archive, name
        return

    if args.archive in ('ucr', 'both'):
        names = ucr_dataset_list()
        if args.limit is not None:
            names = names[: args.limit]
        for name in names:
            yield 'ucr', name

    if args.archive in ('uea', 'both'):
        names = uea_dataset_list()
        if args.limit is not None:
            names = names[: args.limit]
        for name in names:
            yield 'uea', name


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        from pyts.datasets import fetch_ucr_dataset, fetch_uea_dataset
    except ImportError:
        print(
            "Could not import pyts. Install it first, e.g. with "
            "'pip install -e .' from the repository root.",
            file=sys.stderr,
        )
        return 2

    socket.setdefaulttimeout(args.timeout)

    cleanup_dir = None
    if args.data_home is not None:
        data_home_root = Path(args.data_home)
        data_home_root.mkdir(parents=True, exist_ok=True)
    elif args.use_default_cache:
        data_home_root = None  # let fetch_* use its own default
        args.keep = True
    else:
        cleanup_dir = tempfile.mkdtemp(prefix='pyts_dataset_check_')
        data_home_root = Path(cleanup_dir)
        print(f"Downloading into temporary directory: {data_home_root}")

    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = Path(f'dataset_download_report_{timestamp}.csv')
    else:
        report_path = Path(args.output)

    targets = list(_iter_targets(args))
    total = len(targets)
    results = []

    print(f"Checking {total} dataset download(s)...\n")

    try:
        for i, (archive, name) in enumerate(targets, start=1):
            if archive == 'ucr':
                fetch = fetch_ucr_dataset
                data_home = (
                    None
                    if data_home_root is None
                    else str(data_home_root / 'UCR')
                )
            else:
                fetch = fetch_uea_dataset
                data_home = (
                    None
                    if data_home_root is None
                    else str(data_home_root / 'UEA')
                )

            prefix = f"[{i}/{total}] {archive.upper():3s} {name}"
            print(f"{prefix} ... ", end='', flush=True)

            start = time.perf_counter()
            attempt = 0
            backoff = args.retry_delay
            while True:
                attempt += 1
                try:
                    bunch = fetch(
                        name,
                        use_cache=args.use_cache,
                        data_home=data_home,
                        return_X_y=False,
                    )
                    elapsed = time.perf_counter() - start
                    train_shape = tuple(bunch.data_train.shape)
                    test_shape = tuple(bunch.data_test.shape)
                    retry_note = (
                        f", {attempt} attempts" if attempt > 1 else ""
                    )
                    print(f"OK ({elapsed:.1f}s, train={train_shape}, "
                          f"test={test_shape}{retry_note})")
                    results.append({
                        'archive': archive,
                        'dataset': name,
                        'status': 'OK',
                        'elapsed_seconds': f'{elapsed:.2f}',
                        'train_shape': train_shape,
                        'test_shape': test_shape,
                        'error': '',
                    })
                    break
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    if attempt <= args.retries:
                        print(
                            f"retry {attempt}/{args.retries} "
                            f"({error_text}), waiting {backoff:.0f}s... ",
                            end='',
                            flush=True,
                        )
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    elapsed = time.perf_counter() - start
                    print(f"FAILED ({elapsed:.1f}s, {attempt} attempts) "
                          f"- {error_text}")
                    results.append({
                        'archive': archive,
                        'dataset': name,
                        'status': 'FAILED',
                        'elapsed_seconds': f'{elapsed:.2f}',
                        'train_shape': '',
                        'test_shape': '',
                        'error': (
                            error_text + '\n' + traceback.format_exc()
                        ),
                    })
                    break

            if args.delay > 0 and i < total:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted by user, writing partial report...")
    finally:
        if cleanup_dir is not None and not args.keep:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'archive', 'dataset', 'status', 'elapsed_seconds',
                'train_shape', 'test_shape', 'error',
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    n_ok = sum(1 for r in results if r['status'] == 'OK')
    n_failed = sum(1 for r in results if r['status'] == 'FAILED')
    n_run = len(results)

    print(f"\n{'=' * 60}")
    print(f"Checked {n_run}/{total} dataset(s): {n_ok} OK, {n_failed} FAILED")
    print(f"Report written to: {report_path.resolve()}")

    if n_failed > 0:
        print("\nFailed datasets:")
        for r in results:
            if r['status'] == 'FAILED':
                first_line = r['error'].splitlines()[0]
                print(f"  - {r['archive'].upper()} {r['dataset']}: "
                      f"{first_line}")

    return 1 if (n_failed > 0 or n_run < total) else 0


if __name__ == '__main__':
    sys.exit(main())
