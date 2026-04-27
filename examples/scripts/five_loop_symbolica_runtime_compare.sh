#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${HYBRID3D_RUNTIME_DIR:-$ROOT/tmp/five_loop_symbolica_runtime_compare}"
BATCH="${HYBRID3D_PROFILE_BATCH:-100}"
SEED="${HYBRID3D_PROFILE_SEED:-1337}"
N_CORES="${HYBRID3D_SYMBOLICA_N_CORES:-4}"
NO_NATIVE="${HYBRID3D_SYMBOLICA_NO_NATIVE:-0}"

NO_REPEAT_DOT="$ROOT/examples/graphs/five_loop_no_repeats.dot"
REPEATED_DOT="$ROOT/examples/graphs/five_loop_ultimate_basis1.dot"
NO_REPEAT_MASSES='{"mR4a":1.11,"mR4b":1.07,"mR4c":1.19,"mR4d":1.03,"mR3a":0.93,"mR3b":0.89,"mR3c":0.97,"mB1":0.71,"mB2":1.23,"mK1":0.82,"mK2":1.06,"mK3":0.88,"mRet":1.17}'
REPEATED_MASSES='{"mR4":1.11,"mR3":0.93,"mB1":0.71,"mB2":1.23,"mK1":0.82,"mK2":1.06,"mK3":0.88,"mRet":1.17}'

mkdir -p "$OUT"

section() {
  printf '\n==== %s ====\n' "$1"
}

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

compile_extra_args() {
  if [[ "$NO_NATIVE" == "1" ]]; then
    printf '%s\n' "--no-native"
  fi
}

dot_factor() {
  local edge="$1"
  local ext=$((edge % 4))
  printf 'dot(edges[%d], ext[%d])' "$edge" "$ext"
}

csv_contains() {
  local needle="$1"
  local csv="$2"
  local item
  IFS=',' read -r -a items <<< "$csv"
  for item in "${items[@]}"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

numerator_for() {
  local n_edges="$1"
  local squared_csv="${2:-}"
  local expr=""
  local i factor
  for ((i = 0; i < n_edges; i++)); do
    factor="$(dot_factor "$i")"
    if [[ -n "$squared_csv" ]] && csv_contains "$i" "$squared_csv"; then
      factor="($factor)**2"
    else
      factor="($factor)"
    fi
    if [[ -z "$expr" ]]; then
      expr="$factor"
    else
      expr="$expr + $factor"
    fi
  done
  printf '%s\n' "$expr"
}

bounds_for() {
  local squared_csv="${1:-}"
  if [[ -z "$squared_csv" ]]; then
    printf ''
    return
  fi
  local json='{'
  local item
  local sep=''
  IFS=',' read -r -a items <<< "$squared_csv"
  for item in "${items[@]}"; do
    json="$json$sep\"$item\":2"
    sep=','
  done
  json="$json}"
  printf '%s\n' "$json"
}

build_and_compile() {
  local family="$1"
  local dot="$2"
  local numerator="$3"
  local bounds="$4"
  local json="$5"
  local so="$6"
  local build_cmd=(python3 generalised_ltd.py build --family "$family" --dot "$dot" --json-out "$json")
  if [[ -n "$bounds" ]]; then
    build_cmd+=(--energy-degree-bounds "$bounds")
  fi
  run "${build_cmd[@]}"

  local compile_cmd=(python3 generalised_ltd.py compile --orientation-json "$json" --dot "$dot" --numerator-expr "$numerator" --value-type real --output "$so" --n-cores "$N_CORES")
  while IFS= read -r extra; do
    [[ -n "$extra" ]] && compile_cmd+=("$extra")
  done < <(compile_extra_args)
  run "${compile_cmd[@]}"
}

profile_scenario() {
  local case_name="$1"
  local dot="$2"
  local masses="$3"
  local n_edges="$4"
  local scenario="$5"
  local squared_csv="$6"
  local numerator bounds scenario_dir

  scenario_dir="$OUT/$case_name/$scenario"
  mkdir -p "$scenario_dir"

  if [[ "$scenario" == "unit" ]]; then
    numerator="1"
    bounds=""
  else
    numerator="$(numerator_for "$n_edges" "$squared_csv")"
    bounds="$(bounds_for "$squared_csv")"
  fi

  section "$case_name / $scenario"
  printf '%s\n' "$numerator" > "$scenario_dir/numerator.txt"
  [[ -n "$bounds" ]] && printf '%s\n' "$bounds" > "$scenario_dir/energy_degree_bounds.json"

  build_and_compile ltd "$dot" "$numerator" "$bounds" "$scenario_dir/ltd.json" "$scenario_dir/ltd.so"
  build_and_compile cff "$dot" "$numerator" "$bounds" "$scenario_dir/cff.json" "$scenario_dir/cff.so"
  build_and_compile hybrid "$dot" "$numerator" "$bounds" "$scenario_dir/hybrid.json" "$scenario_dir/hybrid.so"

  printf '\n$ python3 generalised_ltd.py evaluate --orientation-json %q --profile-label ltd --profile-json cff=%q --profile-json hybrid=%q --dot %q --masses %q --use-symbolica --profiling %q --seed %q | tee %q\n' \
    "$scenario_dir/ltd.json" \
    "$scenario_dir/cff.json" \
    "$scenario_dir/hybrid.json" \
    "$dot" \
    "$masses" \
    "$BATCH" \
    "$SEED" \
    "$scenario_dir/profile_table.txt"
  python3 generalised_ltd.py evaluate \
    --orientation-json "$scenario_dir/ltd.json" \
    --profile-label ltd \
    --profile-json "cff=$scenario_dir/cff.json" \
    --profile-json "hybrid=$scenario_dir/hybrid.json" \
    --dot "$dot" \
    --masses "$masses" \
    --use-symbolica \
    --profiling "$BATCH" \
    --seed "$SEED" \
    | tee "$scenario_dir/profile_table.txt"

  printf '\n$ python3 generalised_ltd.py evaluate --orientation-json %q --profile-label ltd --profile-json cff=%q --profile-json hybrid=%q --dot %q --masses %q --stability --seed %q --no-color | tee %q\n' \
    "$scenario_dir/ltd.json" \
    "$scenario_dir/cff.json" \
    "$scenario_dir/hybrid.json" \
    "$dot" \
    "$masses" \
    "$SEED" \
    "$scenario_dir/stability_table.txt"
  python3 generalised_ltd.py evaluate \
    --orientation-json "$scenario_dir/ltd.json" \
    --profile-label ltd \
    --profile-json "cff=$scenario_dir/cff.json" \
    --profile-json "hybrid=$scenario_dir/hybrid.json" \
    --dot "$dot" \
    --masses "$masses" \
    --stability \
    --seed "$SEED" \
    --no-color \
    | tee "$scenario_dir/stability_table.txt"
}

profile_case() {
  local case_name="$1"
  local dot="$2"
  local masses="$3"
  local n_edges="$4"
  local squared_csv="$5"

  profile_scenario "$case_name" "$dot" "$masses" "$n_edges" unit ""
  profile_scenario "$case_name" "$dot" "$masses" "$n_edges" linear_all_edges ""
  profile_scenario "$case_name" "$dot" "$masses" "$n_edges" three_squared_edges "$squared_csv"
}

section "Symbolica runtime comparison setup"
printf 'Output directory: %s\n' "$OUT"
printf 'Batch size: %s\n' "$BATCH"
printf 'Symbolica n_cores: %s\n' "$N_CORES"

profile_case five_loop_no_repeats "$NO_REPEAT_DOT" "$NO_REPEAT_MASSES" 13 "0,2,7"
profile_case five_loop_repeated "$REPEATED_DOT" "$REPEATED_MASSES" 13 "0,2,7"

section "Outputs"
printf 'Wrote compiled JSON, shared libraries, numerators, and timing tables under %s\n' "$OUT"
