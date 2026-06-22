#!/usr/bin/env bash
set -euo pipefail

case_dir=$1
abacus_bin=$2
out_dir=$3

mkdir -p "${out_dir}"
repo_dir=$(pwd)
cd "${case_dir}"
export OMP_NUM_THREADS=1

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -lms 500 > "${out_dir}/gpu_usage.csv" &
gpu_pid=$!
trap 'kill "${gpu_pid}" 2>/dev/null || true' EXIT

/usr/bin/time -v nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o "${out_dir}/profile" "${abacus_bin}" > "${out_dir}/abacus_stdout.log" 2> "${out_dir}/nsys_run.log"
nsys stats "${out_dir}/profile.nsys-rep" > "${out_dir}/nsys_stats.txt"
sqlite="${out_dir}/profile.sqlite"
nsys export -t sqlite --force-overwrite=true -o "${sqlite}" "${out_dir}/profile.nsys-rep" > "${out_dir}/nsys_export.log"
python3 "${repo_dir}/tools/perf/analyze_nsys_gpu_gaps.py" "${sqlite}" --min-gap-ms 10 --json "${out_dir}/gaps.json"
