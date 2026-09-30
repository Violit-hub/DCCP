#!/usr/bin/env python3
"""Rebuild the offline visual inspection page at any point."""

import argparse

from _bootstrap import prepare, refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config, require_inputs=False)
    print(refresh(cfg))


if __name__ == "__main__": main()
