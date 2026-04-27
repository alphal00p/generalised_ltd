#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DOT="$ROOT/examples/graphs/box_pow3.dot"
OUT="${HYBRID3D_DEMO_DIR:-$ROOT/tmp/box_pow3_cli_showcase}"
BATCH="${HYBRID3D_PROFILE_BATCH:-100}"
NUMERATOR="${HYBRID3D_NUMERATOR:-dot(edges[0], ext[0]) + edges[3][0]}"
MASSES='{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}'

mkdir -p "$OUT"

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

section() {
  printf '\n==== %s ====\n' "$1"
}

section "Pretty overview of all three representations"
for family in ltd cff hybrid; do
  section "build --family $family --pretty"
  run python3 generalised_ltd.py build \
    --family "$family" \
    --dot "$DOT" \
    --pretty \
    --no-color \
    | tee "$OUT/box_pow3_${family}_pretty.txt"
done

section "Three-way comparison test"
run python3 generalised_ltd.py test \
  --dot "$DOT" \
  --masses "$MASSES" \
  --numerator-expr "$NUMERATOR" \
  --dps 50 \
  --json-out "$OUT/box_pow3_three_way.json"

section "Build JSON representations for evaluation"
run python3 generalised_ltd.py build \
  --family ltd \
  --dot "$DOT" \
  --json-out "$OUT/box_pow3_ltd_builtin.json"

run python3 generalised_ltd.py build \
  --family hybrid \
  --dot "$DOT" \
  --json-out "$OUT/box_pow3_hybrid_builtin.json"

run python3 generalised_ltd.py build \
  --family cff \
  --dot "$DOT" \
  --json-out "$OUT/box_pow3_cff_builtin.json"

run cp "$OUT/box_pow3_ltd_builtin.json" "$OUT/box_pow3_ltd_symbolica_real.json"
run cp "$OUT/box_pow3_hybrid_builtin.json" "$OUT/box_pow3_hybrid_symbolica_real.json"
run cp "$OUT/box_pow3_hybrid_builtin.json" "$OUT/box_pow3_hybrid_symbolica_complex.json"
run cp "$OUT/box_pow3_cff_builtin.json" "$OUT/box_pow3_cff_symbolica_real.json"
run cp "$OUT/box_pow3_cff_builtin.json" "$OUT/box_pow3_cff_symbolica_complex.json"

section "Compile LTD Symbolica evaluator for stability reference"
run python3 generalised_ltd.py compile \
  --orientation-json "$OUT/box_pow3_ltd_symbolica_real.json" \
  --dot "$DOT" \
  --numerator-expr "$NUMERATOR" \
  --value-type real \
  --output "$OUT/box_pow3_ltd_real.so"

section "Compile real-valued Symbolica evaluator and display Symbolica input"
printf '\n$ python3 generalised_ltd.py compile --orientation-json %q --dot %q --numerator-expr %q --value-type real --display-expression --output %q | tee %q\n' \
  "$OUT/box_pow3_hybrid_symbolica_real.json" \
  "$DOT" \
  "$NUMERATOR" \
  "$OUT/box_pow3_hybrid_real.so" \
  "$OUT/box_pow3_hybrid_real_compile_display.txt"
python3 generalised_ltd.py compile \
  --orientation-json "$OUT/box_pow3_hybrid_symbolica_real.json" \
  --dot "$DOT" \
  --numerator-expr "$NUMERATOR" \
  --value-type real \
  --display-expression \
  --output "$OUT/box_pow3_hybrid_real.so" \
  | tee "$OUT/box_pow3_hybrid_real_compile_display.txt"

section "Compile complex-valued Symbolica evaluator"
run python3 generalised_ltd.py compile \
  --orientation-json "$OUT/box_pow3_hybrid_symbolica_complex.json" \
  --dot "$DOT" \
  --numerator-expr "$NUMERATOR" \
  --value-type complex \
  --output "$OUT/box_pow3_hybrid_complex.so"

section "Compile CFF Symbolica evaluators"
run python3 generalised_ltd.py compile \
  --orientation-json "$OUT/box_pow3_cff_symbolica_real.json" \
  --dot "$DOT" \
  --numerator-expr "$NUMERATOR" \
  --value-type real \
  --output "$OUT/box_pow3_cff_real.so"

run python3 generalised_ltd.py compile \
  --orientation-json "$OUT/box_pow3_cff_symbolica_complex.json" \
  --dot "$DOT" \
  --numerator-expr "$NUMERATOR" \
  --value-type complex \
  --output "$OUT/box_pow3_cff_complex.so"

section "Single-value evaluation: built-in vs Symbolica real vs Symbolica complex"
run python3 generalised_ltd.py evaluate \
  --orientation-json "$OUT/box_pow3_hybrid_builtin.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --numerator-expr "$NUMERATOR" \
  --seed 1337 \
  --dps 80

run python3 generalised_ltd.py evaluate \
  --orientation-json "$OUT/box_pow3_hybrid_symbolica_real.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --use-symbolica \
  --seed 1337

run python3 generalised_ltd.py evaluate \
  --orientation-json "$OUT/box_pow3_hybrid_symbolica_complex.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --use-symbolica \
  --seed 1337

section "Profiling: built-in vs Symbolica real vs Symbolica complex"
run python3 generalised_ltd.py evaluate \
  --orientation-json "$OUT/box_pow3_hybrid_builtin.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --numerator-expr "$NUMERATOR" \
  --profiling "$BATCH" \
  --seed 1337 \
  --dps 80 \
  | tee "$OUT/profile_builtin.json"

run python3 generalised_ltd.py evaluate \
  --orientation-json "$OUT/box_pow3_hybrid_symbolica_real.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --use-symbolica \
  --profiling "$BATCH" \
  --seed 1337 \
  | tee "$OUT/profile_symbolica_real.json"

run python3 generalised_ltd.py evaluate \
  --orientation-json "$OUT/box_pow3_hybrid_symbolica_complex.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --use-symbolica \
  --profiling "$BATCH" \
  --seed 1337 \
  | tee "$OUT/profile_symbolica_complex.json"

section "Profiling table: all Symbolica evaluator modes"
printf '\n$ python3 generalised_ltd.py evaluate --orientation-json %q --profile-label hybrid-real --profile-json hybrid-complex=%q --profile-json cff-real=%q --profile-json cff-complex=%q --dot %q --masses %q --evaluator-backend symbolica --profiling %q --seed 1337 --no-color | tee %q\n' \
  "$OUT/box_pow3_hybrid_symbolica_real.json" \
  "$OUT/box_pow3_hybrid_symbolica_complex.json" \
  "$OUT/box_pow3_cff_symbolica_real.json" \
  "$OUT/box_pow3_cff_symbolica_complex.json" \
  "$DOT" \
  "$MASSES" \
  "$BATCH" \
  "$OUT/profile_table.txt"
python3 generalised_ltd.py evaluate \
  --orientation-json "$OUT/box_pow3_hybrid_symbolica_real.json" \
  --profile-label hybrid-real \
  --profile-json "hybrid-complex=$OUT/box_pow3_hybrid_symbolica_complex.json" \
  --profile-json "cff-real=$OUT/box_pow3_cff_symbolica_real.json" \
  --profile-json "cff-complex=$OUT/box_pow3_cff_symbolica_complex.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --evaluator-backend symbolica \
  --profiling "$BATCH" \
  --seed 1337 \
  --no-color \
  | tee "$OUT/profile_table.txt"

section "Stability table: LTD vs CFF vs hybrid"
printf '\n$ python3 generalised_ltd.py evaluate --orientation-json %q --profile-label ltd --profile-json cff=%q --profile-json hybrid=%q --dot %q --masses %q --stability --seed 1337 --no-color | tee %q\n' \
  "$OUT/box_pow3_ltd_symbolica_real.json" \
  "$OUT/box_pow3_cff_symbolica_real.json" \
  "$OUT/box_pow3_hybrid_symbolica_real.json" \
  "$DOT" \
  "$MASSES" \
  "$OUT/stability_table.txt"
python3 generalised_ltd.py evaluate \
  --orientation-json "$OUT/box_pow3_ltd_symbolica_real.json" \
  --profile-label ltd \
  --profile-json "cff=$OUT/box_pow3_cff_symbolica_real.json" \
  --profile-json "hybrid=$OUT/box_pow3_hybrid_symbolica_real.json" \
  --dot "$DOT" \
  --masses "$MASSES" \
  --stability \
  --seed 1337 \
  --no-color \
  | tee "$OUT/stability_table.txt"

section "Outputs"
printf 'Wrote demo artifacts to %s\n' "$OUT"
