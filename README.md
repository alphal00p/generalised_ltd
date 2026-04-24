# hybrid3d DOT-only prototype

This package is a DOT-graph-first prototype for inspecting and numerically evaluating LTD, CFF and hybrid LTD/CFF orientation structures for graphs with repeated propagator channels.

## Graph convention

Internal propagators are directed DOT edges carrying a linear momentum label such as `k1+p1` and an optional symbolic mass key:

```dot
v0 -> v1 [label="k1+p1", mass="mA"];
```

If `mass` is absent the edge is massless.  Equal symbolic mass keys indicate equal masses at evaluation time.  Numeric masses are supplied separately with `--masses`.  The edge attribute `pow` is not supported; raised propagators must be represented by repeated distinct edges.

Repeated propagators must not be encoded as duplicate parallel edges.  They should be serialized through intermediate vertices or through the sandwiched bubble structure described in the examples.

## Commands

Validate a graph:

```bash
python hybrid3d.py validate --dot examples/box_pow3.dot
```

Build and inspect the hybrid structure:

```bash
python hybrid3d.py build --family hybrid --dot examples/box_pow3.dot --pretty
python hybrid3d.py build --family hybrid --dot examples/box_pow3.dot --pretty --show-details-for-orientation '000+--|+'
```

The orientation label before `|` is the edge-orientation string on the original internal edges.  The marker after `|` is the chain sign for each repeated channel group, ordered by the minimum edge id in each repeated group.

Export JSON:

```bash
mkdir -p demo
python hybrid3d.py build --family hybrid --dot examples/box_pow3.dot --json-out demo/box_hybrid.json
```

Evaluate from JSON:

```bash
python hybrid3d.py evaluate \
  --orientation-json demo/box_hybrid.json \
  --dot examples/box_pow3.dot \
  --external '[[0.3,0.1,-0.2,0.05],[-0.15,0.2,0.05,-0.1],[0.25,-0.1,0.15,0.07]]' \
  --loop3 '[[0.2,-0.3,0.1]]' \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}' \
  --numerator-expr 'dot(edges[0], ext[0]) + dot(edges[3], ext[0])'
```

Run the diagnostic three-way report:

```bash
python hybrid3d.py test --dot examples/proper_iterated_sandwiched_bubble.dot \
  --masses '{"mA":0.8,"mB":0.9,"mC":1.1,"mD":0.75,"mE":1.0,"mF":0.6}' \
  --numerator-expr 'dot(edges[1], ext[0]) + dot(edges[4], ext[0])'
```

## JSON structure

The JSON is intentionally DOT-dependent.  It stores:

- the family and backend,
- the graph summary,
- a surface cache with compact linear expressions in `E[i]` and `OSE[a]`,
- one entry per orientation,
- the per-orientation loop and edge energy substitutions.

`OSE[a]` is now always an external basis-energy id, not an external half-edge id.  Therefore a box with three external basis vectors only displays `OSE[0]`, `OSE[1]` and `OSE[2]`.

## Pretty output

The pretty output is meant for auditing.  By default it shows only the graph summary, surface cache and orientation summary.  Use `--show-details-for-orientation LABEL` to inspect one orientation, including the full substitution maps and the factorization tree.

## Tests

Run:

```bash
pytest -q tests
```

In this environment the packaged test suite was audited with the Python command passed through `HYBRID3D_PYTHON`, for example:

```bash
HYBRID3D_PYTHON='/opt/pyvenv/bin/python -S' \
PYTHONPATH=/opt/pyvenv/lib/python3.13/site-packages:. \
/opt/pyvenv/bin/python -S -m pytest -q tests
```

The tests emphasize graph validation, family-distinct structures, JSON-driven evaluation, mutation sensitivity of substitution maps and skeleton surfaces, edge-by-edge dot-product numerator checks, and three-way CFF/hybrid/split-mass LTD diagnostics on every valid example topology.

## Current status

The package is an actively evolving research prototype.  The current hybrid JSON tree includes the LTD skeleton surfaces as evaluated denominator factors and stores repeated-edge substitution maps separately.  See `docs/hybrid_current_status.pdf` for the theoretical and implementation summary.


## v12 status note

This package now uses the following display convention consistently:

- `OSE[i]` denotes the internal on-shell energy of internal edge `i`.
- `E[a]` denotes the external-energy component of external basis momentum `a`.

For CFF surfaces, external shifts are inferred from the internal momentum routing crossing the contracted vertex set.  The dashed external half-edges in the DOT graph are treated as validation/bookkeeping information and are not directly summed when constructing CFF surface shifts; this avoids the spurious doubled external-energy coefficients that appeared in earlier versions.

The `test` subcommand reports a three-way numerical diagnostic.  It now always evaluates the exported CFF JSON and hybrid JSON directly, and reports the split-mass LTD sequence as an independent limiting reference.  Optional `--cff-json` and `--hybrid-json` inputs can be supplied to make the diagnostic use prebuilt JSON instead of rebuilding both structures.

The CFF tree generator expands all acyclic contractions of the selected source/sink boundary, so non-simplicial orientations are represented as genuine sums in the JSON tree.  For one-loop repeated-channel graphs the hybrid backend uses the fixed-tau LTD/local-cone construction.  For multi-loop repeated-channel graphs the repeated-sector tau closures are coupled; the backend therefore uses the coupled CFF-cone kernel with a separate hybrid orientation namespace rather than the invalid factorized local-cone formula.  Graphs without repeated channels still collapse to the ordinary LTD bundle.
