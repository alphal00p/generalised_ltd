# hybrid3d DOT prototype

`hybrid3d.py` builds, prints, exports, and evaluates JSON representations for
LTD, pure CFF, and the current hybrid LTD/CFF construction for repeated
propagator channels.

Raised propagators are encoded as repeated internal DOT edges with the same
momentum signature and mass key.  The DOT `pow` attribute is intentionally
rejected; use explicit repeated edges instead.

## CLI

Validate a graph:

```bash
python3 hybrid3d.py validate --dot examples/graphs/box_pow3.dot
```

Build a structure:

```bash
python3 hybrid3d.py build --family ltd --dot examples/graphs/box.dot --pretty
python3 hybrid3d.py build --family cff --dot examples/graphs/box.dot --pretty
python3 hybrid3d.py build --family hybrid --dot examples/graphs/box_pow3.dot --pretty
```

Build a DOT graph from a propagator-signature expression:

```bash
python3 hybrid3d.py graph_from_signatures \
  --signatures 'prop(k1+p1,mA)*prop(k1+p1-q1,mB)*prop(k1-p2+q2,mC)*prop(k1,mD)' \
  --dot-output demo/from_signatures.dot
```

The default output uses the same DOT conventions as the examples.  Use
`--format vakint` for the compact single-`ext` node format with ordered edge ids
and `lmb_id` attributes.

Runnable CLI examples live in `examples/scripts`.  The main showcase for the
repeated box topology is:

```bash
examples/scripts/box_pow3_cli_showcase.sh
```

It prints pretty overviews for LTD, CFF, and hybrid structures, runs the
three-way comparison, compiles real and complex Symbolica evaluators, and
profiles built-in versus compiled evaluation.

The five-loop runtime comparison script builds and compiles LTD, CFF, and
hybrid evaluators for a non-repeated four-external five-loop graph and for the
existing repeated five-loop graph:

```bash
examples/scripts/five_loop_symbolica_runtime_compare.sh
```

For each graph it profiles three numerator choices: `1`, a sum with one
dot-product term for every internal edge, and the same sum with three selected
edge terms squared.  The squared-edge scenario intentionally exercises the
bounded-degree builders and can take substantially longer to compile than the
timed evaluation itself.

Inspect one orientation in detail:

```bash
python3 hybrid3d.py build --family cff --dot examples/graphs/box.dot \
  --energy-degree-bounds 0:2,1:2,2:2 \
  --pretty --show-details-for-orientation --++ --no-color
```

Export and evaluate JSON:

```bash
python3 hybrid3d.py build --family hybrid --dot examples/graphs/box_pow3.dot \
  --json-out demo/box_hybrid.json

python3 hybrid3d.py evaluate \
  --orientation-json demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --external '[[0.3,0.1,-0.2,0.05],[-0.15,0.2,0.05,-0.1],[0.25,-0.1,0.15,0.07]]' \
  --loop3 '[[0.2,-0.3,0.1]]' \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}' \
  --numerator-expr 'dot(edges[0], ext[0]) + dot(edges[3], ext[0])'
```

Compile a fixed-numerator Symbolica evaluator and use it:

```bash
python3 hybrid3d.py compile \
  --orientation-json demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --numerator-expr 'dot(edges[0], ext[0]) + dot(edges[3], ext[0])' \
  --value-type real \
  --display-expression

python3 hybrid3d.py evaluate \
  --orientation-json demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --use-symbolica \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}'

python3 hybrid3d.py evaluate \
  --orientation-json demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --use-symbolica \
  --profiling 10000 \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}'
```

Compare several compiled JSON evaluators in one timing table:

```bash
python3 hybrid3d.py evaluate \
  --orientation-json demo/box_ltd.json \
  --profile-label ltd \
  --profile-json cff=demo/box_cff.json \
  --profile-json hybrid=demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --use-symbolica \
  --profiling 10000 \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}'
```

The `compile` command stores an `evaluator` block in the JSON and writes a
shared-library evaluator (`.so`) next to it by default.  The library path is
stored relative to the JSON file.  The compiled evaluator is tied to the exact
DOT graph, JSON structure, value type, and numerator expression.  If
`SYMBOLICA_LICENSE` is set in the environment, Symbolica can use multicore
optimization during compilation.
`--display-expression` prints the Symbolica top-level expression, parameter
order, constants, and function map before compilation.

Run diagnostics:

```bash
python3 hybrid3d.py test --dot examples/graphs/proper_iterated_sandwiched_bubble.dot \
  --masses '{"mA":0.8,"mB":0.9,"mC":1.1,"mD":0.75,"mE":1.0,"mF":0.6}' \
  --numerator-expr 'dot(edges[1], ext[0]) + dot(edges[4], ext[0])'

python3 hybrid3d.py test-cff-ltd --dot examples/graphs/box.dot \
  --energy-degree-bounds 0:2,1:2,2:2 \
  --numerator-expr 'edges[0][0]**2 * edges[1][0]**2 * edges[2][0]**2' \
  --dps 80
```

## JSON Semantics

The exported schema is intentionally evaluator-complete.  Evaluation uses only:

- the DOT graph,
- the JSON surface cache,
- the orientation energy maps,
- half-edge factors,
- numerator-side surface factors,
- and the serialized factorization trees.

There is no hidden LTD/CFF backend call during evaluation.

If the optional top-level `evaluator` block is present, it describes a compiled
Symbolica evaluator for one fixed numerator.  The ordinary JSON semantics remain
authoritative: `evaluate --use-symbolica` only changes how that same serialized
orientation sum is numerically evaluated.

Schema v6 uses this invariant:

> one orientation equals one unique EMR edge-energy numerator map.

All denominator contributions with that same numerator call are stored under
`variants`.  Variant metadata records physical origin such as `cff`, `ltd`,
`pinch[...]`, or hybrid basis/interpolation labels.  The pretty table therefore
has one orientation id and possibly many variant rows.

Pretty labels use:

- `+`: sample edge `e` at `+OSE[e]`,
- `-`: sample edge `e` at `-OSE[e]`,
- `0`: sample edge `e` at zero energy,
- `x`: a non-trivial linear map, shown in detail with `--show-details`.

Surface classes are printed as `e` or `h`; numerator-only surfaces are printed
as `(e)` or `(h)`.

## Energy-Degree Bounds

`--energy-degree-bounds` supplies an upper bound for the EMR energy degree of
each edge in the numerator:

```bash
python3 hybrid3d.py build --family cff --dot examples/graphs/box.dot \
  --energy-degree-bounds 0:2,1:2,2:2 --pretty --no-color
```

The builder checks both coordinate loop-energy UV degree and every
one-parameter direction in loop-energy space.  If a residue at infinity may
contribute, the build fails.

Current exact bounded-degree support:

- `ltd`: no structural change; bounds are reported only.
- `hybrid` without repeated propagators: collapses to LTD even with bounds.
- `hybrid` with repeated propagators: bounded numerators are handled by the
  confluent hybrid formula with finite-difference numerator samples in the
  residue-basis energy coordinates.
- `cff`: one-loop non-repeated graphs support arbitrary quadratic-or-lower caps
  with only regular `E`-surfaces in denominators.
- `cff`: multiloop and split-repeated graphs support quadratic-or-lower caps by
  summing finite-pole remainder/contact sectors.  Deleted lower sectors are
  decomposed into loop-energy matroid components and rebuilt as auxiliary CFF
  causal minors, so denominator surfaces remain `E`-only; numerator-side cached
  surfaces may still be `E` or `H`.
- `cff`: unsplit repeated-propagator graphs with quadratic caps use the
  confluent repeated-pole limit also used by bounded `hybrid`, avoiding singular
  equal-mass contact denominators.
- `cff`: isolated single-edge cubic caps are supported.

Unsupported bounded CFF cases raise `NotImplementedError`; the old
`CFF + (LTD - CFF)` contact fallback has been removed.

The main open distinction is quadratic versus genuinely higher power.  Quadratic
contacts produce only the three black-box samples `+E`, `0`, `-E` and no known
polynomial numerator residue, so the implemented path covers the tested
quadratic caps, including multiloop energy monomials and individual squared
edge--external dot products.  Products that make a deleted lower causal
component quadratic through several different spectator edge energies are the
next refinement: they require interpolation in a component energy basis, not
only per-edge finite-pole divisions.  Cubic, quartic, and mixed higher caps
produce known numerator-side polynomial factors after a pinch; those factors
must be recursively reduced before the result can again be serialized.

## Tests

Default suite:

```bash
PYTHONPATH=. pytest -q tests
```

The default suite includes:

- validation for all example DOT graphs except the intentionally invalid noisy
  example,
- JSON evaluator mutation tests,
- one-numerator-call-per-orientation tests,
- CFF/LTD/hybrid structural checks,
- repeated-propagator three-way diagnostics,
- bounded-degree CFF/LTD checks for supported caps,
- Symbolica compiled-evaluator checks when the optional package is installed,
- a non-repeated four-external five-loop graph,
- a four-loop repeated-channel stress topology.

The old five-loop ultimate basis alignment test is still present but slow.  Run
it explicitly with:

```bash
HYBRID3D_RUN_SLOW=1 PYTHONPATH=. pytest -q tests/test_cli.py::test_ultimate_five_loop_bases_three_way_and_aligned_momenta
```

See `docs/hybrid_3d.pdf` for the derivation, JSON mapping, and current bounded
degree limitations.
