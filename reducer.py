#!/usr/bin/env python3
import sys

# Initialize variables
current_key = None
current_count = 0

# Read lines from standard input
for line in sys.stdin:
    line = line.strip()
    # Example aggregation logic
    if current_key == line:
        current_count += 1
    else:
        if current_key:
            print(f"{current_key}\t{current_count}")  # Output the previous key and count
        current_key = line
        current_count = 1

# Output the last key
if current_key == line:
    print(f"{current_key}\t{current_count}")
