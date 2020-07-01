#!/usr/bin/env bash

for filepath in ./config/*.json; do
	echo "Executing hyriseBenchmarkTPCH with encoding/compression config $filepath"
	mkdir -p ./benchmark_output
    $HOME/hyrise/cmake-build-debug/hyriseBenchmarkTPCH -r "100" -o "./benchmark_output/output_$(basename "$filepath" .json).json" -e "$filepath" --dont_cache_binary_tables
done
