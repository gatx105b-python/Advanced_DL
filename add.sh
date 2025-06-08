#!/bin/bash

MAX_SIZE=1048576  # 1MB = 1024 * 1024

find parity_comp -type f | while read file; do
    size=$(stat -c%s "$file")
    if [ "$size" -le "$MAX_SIZE" ]; then
        git add -f "$file"  # -f: .gitignore에 무시돼도 강제 추가
        echo "추가됨: $file"
    else
        echo "건너뜀 (1MB 초과): $file"
    fi
done
