#!/bin/bash

# Define the directories to search
directories=("include" "source" "test" "standalone")

# Find all .cpp, .cu, and .h files in the specified directories
src_files=$(find "${directories[@]}" -type f \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" \))

# Check if any files are found
if [ -z "$src_files" ]; then
    echo "No files to format."
    exit 0
fi

# Format each file using clang-format
for file in ${src_files}; do
    echo "Formatting ${file}"
    clang-format -i "${file}"
done

echo "Buddy, well done!"