#!/bin/bash
# Checking if the required argument is provided
if [ $# -ne 1 ]; then
    echo "Input should be of type: $0 <filename>"
    exit 1
fi
# Get the input filename from the command line argument
file="$1"
# Check if the file has a ".npy" extension
if [[ "$file" != *.npy ]]; then
    echo "Error: File extension must be '.npy'."
    exit 1
fi
# Check if the file exists in the current directory
if [ -e "$file" ]; then
# Run test.py with the provided filename as an argument
    python3 test.py "$file"
else
    echo "Error: File '$file' not found in the current directory."
    exit 1
fi