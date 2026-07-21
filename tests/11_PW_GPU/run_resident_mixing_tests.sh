#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 CASE_LIST TAU_CASE [REQUIRED_MARKER]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

case_list=$1
tau_case=$2
required_marker=${3:-}

bash ../integrate/Autotest.sh -n 1 -t 1e-3 -f "$case_list"

while IFS= read -r case_name; do
    running_log="${case_name}/OUT.autotest/running_scf.log"
    case_log="${case_name}/log.txt"

    grep -q 'INFO: Using GPU-resident reciprocal charge mixing.' "$running_log"
    if [[ -n "$required_marker" ]]; then
        grep -qF "$required_marker" "$running_log"
    fi
    awk '/^ CG/ {
           rows++
           for (i = 2; i <= NF; ++i) {
             value = tolower($i)
             if (value ~ /^[+-]?(nan|inf|infinity)$/) bad = 1
           }
         }
         END { exit !(rows == 3 && !bad) }' "$case_log"
    awk '/Number of plane waves =/ { values[++count] = $NF }
         END { exit !(count >= 2 && values[1] != values[2]) }' "$running_log"
done < "$case_list"

grep -q 'DKIN' "${tau_case}/log.txt"
