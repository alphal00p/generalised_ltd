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
python3 hybrid3d.py validate --dot examples/box_pow3.dot
```

Build a structure:

```bash
python3 hybrid3d.py build --family ltd --dot examples/box.dot --pretty
python3 hybrid3d.py build --family cff --dot examples/box.dot --pretty
python3 hybrid3d.py build --family hybrid --dot examples/box_pow3.dot --pretty
```

Inspect one orientation in detail:

```bash
python3 hybrid3d.py build --family cff --dot examples/box.dot \
  --energy-degree-bounds 0:2,1:2,2:2 \
  --pretty --show-details-for-orientation --++ --no-color
```

Export and evaluate JSON:

```bash
python3 hybrid3d.py build --family hybrid --dot examples/box_pow3.dot \
  --json-out demo/box_hybrid.json

python3 hybrid3d.py evaluate \
  --orientation-json demo/box_hybrid.json \
  --dot examples/box_pow3.dot \
  --external '[[0.3,0.1,-0.2,0.05],[-0.15,0.2,0.05,-0.1],[0.25,-0.1,0.15,0.07]]' \
  --loop3 '[[0.2,-0.3,0.1]]' \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}' \
  --numerator-expr 'dot(edges[0], ext[0]) + dot(edges[3], ext[0])'
```

Run diagnostics:

```bash
python3 hybrid3d.py test --dot examples/proper_iterated_sandwiched_bubble.dot \
  --masses '{"mA":0.8,"mB":0.9,"mC":1.1,"mD":0.75,"mE":1.0,"mF":0.6}' \
  --numerator-expr 'dot(edges[1], ext[0]) + dot(edges[4], ext[0])'

python3 hybrid3d.py test-cff-ltd --dot examples/box.dot \
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
python3 hybrid3d.py build --family cff --dot examples/box.dot \
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
- a four-loop repeated-channel stress topology.

The old five-loop ultimate basis alignment test is still present but slow.  Run
it explicitly with:

```bash
HYBRID3D_RUN_SLOW=1 PYTHONPATH=. pytest -q tests/test_cli.py::test_ultimate_five_loop_bases_three_way_and_aligned_momenta
```

See `docs/hybrid_3d.pdf` for the derivation, JSON mapping, and current bounded
degree limitations.
