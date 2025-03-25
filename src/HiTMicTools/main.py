#!/usr/bin/env python3
import sys

sys.path.insert(0, "./src")

import sys
from HiTMicTools.cli import hitmictools


def main():
    """Main entry point for HiTMicTools CLI"""
    parser = hitmictools()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    # Use the CLI parser
    main()
