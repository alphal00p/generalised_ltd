#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DOT="$ROOT/examples/graphs/one_loop_10_external.dot"
OUT="${HYBRID3D_10EXT_RUNTIME_DIR:-$ROOT/tmp/one_loop_10_external_runtime_compare}"
BATCH="${HYBRID3D_PROFILE_BATCH:-100}"
SEED="${HYBRID3D_PROFILE_SEED:-1337}"
N_CORES="${HYBRID3D_SYMBOLICA_N_CORES:-4}"
NO_NATIVE="${HYBRID3D_SYMBOLICA_NO_NATIVE:-0}"
MASSES='{"m1":0.60,"m2":0.68,"m3":0.76,"m4":0.84,"m5":0.92,"m6":1.00,"m7":1.08,"m8":1.16,"m9":1.24,"m10":1.32}'

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

build_and_compile() {
  local family="$1"
  local json="$OUT/${family}.json"
  local so="$OUT/${family}.so"
  local compile_cmd

  run python3 hybrid3d.py build \
    --family "$family" \
    --dot "$DOT" \
    --json-out "$json"

  compile_cmd=(python3 hybrid3d.py compile \
    --orientation-json "$json" \
    --dot "$DOT" \
    --numerator-expr "1" \
    --value-type real \
    --output "$so" \
    --n-cores "$N_CORES")
  while IFS= read -r extra; do
    [[ -n "$extra" ]] && compile_cmd+=("$extra")
  done < <(compile_extra_args)
  run "${compile_cmd[@]}"
}

section "One-loop 10-external LTD/CFF Symbolica runtime comparison"
printf 'Output directory: %s\n' "$OUT"
printf 'Batch size: %s\n' "$BATCH"
printf 'Symbolica n_cores: %s\n' "$N_CORES"
printf 'Numerator: 1\n'
printf 'Topology: 10 propagators, 10 external momentum symbols\n'
printf 'Note: the CFF build has many one-loop sign sectors and can be substantially slower than LTD.\n'

build_and_compile ltd
build_and_compile cff

section "Profile all available Symbolica evaluator modes"
printf '\n$ python3 hybrid3d.py evaluate --orientation-json %q --profile-label ltd --profile-json cff=%q --dot %q --masses %q --evaluator-backend symbolica --profiling %q --seed %q | tee %q\n' \
  "$OUT/ltd.json" \
  "$OUT/cff.json" \
  "$DOT" \
  "$MASSES" \
  "$BATCH" \
  "$SEED" \
  "$OUT/profile_table.txt"
python3 hybrid3d.py evaluate \
  --orientation-json "$OUT/ltd.json" \
  --profile-label ltd \
  --profile-json "cff=$OUT/cff.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --evaluator-backend symbolica \
  --profiling "$BATCH" \
  --seed "$SEED" \
  | tee "$OUT/profile_table.txt"

section "Precision stability with Symbolica eager evaluators"
printf '\n$ python3 hybrid3d.py evaluate --orientation-json %q --profile-label ltd --profile-json cff=%q --dot %q --masses %q --stability --seed %q --no-color | tee %q\n' \
  "$OUT/ltd.json" \
  "$OUT/cff.json" \
  "$DOT" \
  "$MASSES" \
  "$SEED" \
  "$OUT/stability_table.txt"
python3 hybrid3d.py evaluate \
  --orientation-json "$OUT/ltd.json" \
  --profile-label ltd \
  --profile-json "cff=$OUT/cff.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --stability \
  --seed "$SEED" \
  --no-color \
  | tee "$OUT/stability_table.txt"

section "Outputs"
printf 'Wrote compiled JSON, eager evaluator state, shared libraries, and timing table under %s\n' "$OUT"
