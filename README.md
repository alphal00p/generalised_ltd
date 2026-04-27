# generalised_ltd DOT toolkit

`generalised_ltd.py` builds, prints, exports, and evaluates JSON representations for
LTD, pure CFF, and the current hybrid LTD/CFF construction for repeated
propagator channels.

Raised propagators are encoded as repeated internal DOT edges with the same
momentum signature and mass key.  Use explicit repeated edges rather than the
DOT `pow` attribute.

## CLI

Validate a graph:

```bash
python3 generalised_ltd.py validate --dot examples/graphs/box_pow3.dot
```

Build a structure:

```bash
python3 generalised_ltd.py build --family ltd --dot examples/graphs/box.dot --pretty
python3 generalised_ltd.py build --family cff --dot examples/graphs/box.dot --pretty
python3 generalised_ltd.py build --family hybrid --dot examples/graphs/box_pow3.dot --pretty
```

Build a DOT graph from a propagator-signature expression:

```bash
python3 generalised_ltd.py graph_from_signatures \
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
profiles built-in versus compiled Symbolica evaluation.

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

The one-loop many-external benchmark compares LTD and CFF for a numerator-free
one-loop 10-gon with 10 propagators and 10 external momentum symbols:

```bash
examples/scripts/one_loop_10_external_runtime_compare.sh
```

The analogous larger benchmark uses a numerator-free
one-loop 15-gon with 15 propagators and 15 external momentum symbols:

```bash
examples/scripts/one_loop_15_external_runtime_compare.sh
```

This script builds real-valued Symbolica evaluators for the two representations
and profiles every Symbolica evaluator mode stored in the JSON.

Inspect one orientation in detail:

```bash
python3 generalised_ltd.py build --family cff --dot examples/graphs/box.dot \
  --energy-degree-bounds 0:2,1:2,2:2 \
  --pretty --show-details-for-orientation --++ --no-color
```

Export and evaluate JSON:

```bash
python3 generalised_ltd.py build --family hybrid --dot examples/graphs/box_pow3.dot \
  --json-out demo/box_hybrid.json

python3 generalised_ltd.py evaluate \
  --orientation-json demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --external '[[0.3,0.1,-0.2,0.05],[-0.15,0.2,0.05,-0.1],[0.25,-0.1,0.15,0.07]]' \
  --loop3 '[[0.2,-0.3,0.1]]' \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}' \
  --numerator-expr 'dot(edges[0], ext[0]) + dot(edges[3], ext[0])'
```

Compile a fixed-numerator Symbolica evaluator and use it:

```bash
python3 generalised_ltd.py compile \
  --orientation-json demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --numerator-expr 'dot(edges[0], ext[0]) + dot(edges[3], ext[0])' \
  --value-type real \
  --display-expression

python3 generalised_ltd.py evaluate \
  --orientation-json demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --evaluator-backend symbolica_compiled \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}'

python3 generalised_ltd.py evaluate \
  --orientation-json demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --evaluator-backend symbolica_eager_symjit \
  --profiling 100 \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}'
```

Profiling times 10 evaluator calls over the requested batch size and reports
both per-sample and per-call timings.  `--profiling` without an explicit value
uses a batch size of 100.

Compare several compiled JSON evaluators in one timing table:

```bash
python3 generalised_ltd.py evaluate \
  --orientation-json demo/box_ltd.json \
  --profile-label ltd \
  --profile-json cff=demo/box_cff.json \
  --profile-json hybrid=demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --evaluator-backend symbolica \
  --profiling 100 \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}'
```

Monitor decimal precision loss with the saved Symbolica eager evaluator:

```bash
python3 generalised_ltd.py evaluate \
  --orientation-json demo/box_ltd.json \
  --profile-label ltd \
  --profile-json cff=demo/box_cff.json \
  --profile-json hybrid=demo/box_hybrid.json \
  --dot examples/graphs/box_pow3.dot \
  --stability \
  --masses '{"m1":0.8,"m2":1.1,"m3":0.9,"m4":1.2}'
```

`--stability` uses the serialized Symbolica eager evaluator, converts the real
input kinematics to precision-tracking decimal floats with 16 significant
digits by default, evaluates at 80 decimal digits of work precision, and reports
the number of significant digits carried by the final result.  Pass
`--stability 20 --stability-work-precision 100` to change those two numbers.

The `compile` command stores an `evaluator` compatibility block plus an
`evaluators` block in the JSON.  It writes both a shared-library evaluator
(`.so`) and a serialized eager Symbolica evaluator (`.sev`) next to the JSON by
default.  `evaluate --evaluator-backend` can select `symbolica_compiled`,
`symbolica_eager`, or `symbolica_eager_symjit`; `symbolica` is an alias for
compiled evaluation in a single run and means all available Symbolica modes in
the multi-JSON profiling table.  The evaluator paths are stored relative to the
JSON file and are tied to the exact DOT graph, JSON structure, value type, and
numerator expression.  If `SYMBOLICA_LICENSE` is set in the environment,
Symbolica can use multicore optimization during evaluator construction.
`--display-expression` prints the Symbolica top-level expression, parameter
order, constants, and function map before compilation.

Run diagnostics:

```bash
python3 generalised_ltd.py test --dot examples/graphs/proper_iterated_sandwiched_bubble.dot \
  --masses '{"mA":0.8,"mB":0.9,"mC":1.1,"mD":0.75,"mE":1.0,"mF":0.6}' \
  --numerator-expr 'dot(edges[1], ext[0]) + dot(edges[4], ext[0])'

python3 generalised_ltd.py test --dot examples/graphs/box_pow3.dot \
  --energy-degree-bounds 0:1,1:1,2:0,3:4 \
  --numerator-expr 'edges[0][0] * edges[1][0] * edges[3][0]**4' \
  --dps 80

python3 generalised_ltd.py test-cff-ltd --dot examples/graphs/box.dot \
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

For each variant, `half_edges` is an evaluated multiset, not debug metadata.
Every occurrence of edge `e` multiplies the variant by `(2*E[e])^-1`; repeated
ids encode powers of the same on-shell-energy residue factor.  The full variant
factor is `pref * prod_e (2*E[e])^-count(e)` times numerator-side surfaces, the
factorized denominator tree, and the numerator sampled at the orientation
energy map.

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
as `(e)` or `(h)`. Causal denominator surfaces from the graph or a lower-sector
CFF component remain physical `e`/`h` surfaces. Interpolation or reconstruction
helpers add `_h`, and helper factors that are not original causal surfaces add
`_sp`, for example `(e_h_sp)`.

## Energy-Degree Bounds

`--energy-degree-bounds` supplies an upper bound for the EMR energy degree of
each edge in the numerator:

```bash
python3 generalised_ltd.py build --family cff --dot examples/graphs/box.dot \
  --energy-degree-bounds 0:2,1:2,2:2 --pretty --no-color
```

The builder checks both coordinate loop-energy UV degree and every
one-parameter direction in loop-energy space.  If a residue at infinity may
contribute, the build fails.

For `test` and `compare`, `--energy-degree-bounds` is the common bounded-degree
configuration used for the CFF expression, the hybrid expression, and the
split-mass LTD reference.  `--cff-energy-degree-bounds` is kept only for
CFF-only diagnostics.

Subcommands that evaluate a numerator (`evaluate`, `compile`, `compare`,
`test`, and `test-cff-ltd`) accept `--numerator-expr auto`.  `build` accepts
the same option as metadata and records the resolved expression in
`graph.numerator_expr` without changing the numerator-independent orientation
structure.  With
`--energy-degree-bounds`, this expands to a product of edge-external dot
products that saturates the requested EMR energy degree on every bounded edge.
Unspecified edges receive degree zero and contribute no factor.  For example,
on `box_pow3.dot`,

```bash
python3 generalised_ltd.py test --dot examples/graphs/box_pow3.dot \
  --energy-degree-bounds 3:3,4:4 --numerator-expr auto
```

uses

```text
dot(edges[3], ext[0]) * dot(edges[3], ext[1]) * dot(edges[3], ext[2]) *
dot(edges[4], ext[1]) * dot(edges[4], ext[2]) * dot(edges[4], ext[0]) *
dot(edges[4], ext[1])
```

as the actual numerator expression reported in the JSON diagnostic.  For
`evaluate` and `compile`, `auto` reads the normalized bounds from
`graph.energy_degree_bounds` in the orientation JSON.

Current exact bounded-degree support:

- `ltd`: no structural change; bounds are reported only.
- `hybrid` without repeated propagators: collapses to LTD even with bounds.
- `hybrid` with repeated propagators: bounded numerators are handled by the
  confluent hybrid formula with finite-difference numerator samples in the
  residue-basis energy coordinates.  Cubic, quartic, and quintic caps are
  covered by tests against the split-mass LTD limiting proxy; larger caps are
  accepted subject to the UV check and practical expression size.
- `cff`: one-loop graphs support arbitrary UV-convergent caps
  with only regular `E`-surfaces in denominators.
- `cff`: multiloop and split-repeated graphs support quadratic-or-lower caps by
  summing finite-pole remainder/contact sectors.  Deleted lower sectors are
  decomposed into loop-energy matroid components and rebuilt as auxiliary CFF
  causal minors, so denominator surfaces remain `E`-only; numerator-side cached
  surfaces may still be `E` or `H`.
- `cff`: cubic, mixed cubic/quadratic, and quartic-or-higher caps are supported
  through a channel normal form for repeated signatures and a recursive
  lower-contact completion for ordinary high-power edges.  This includes
  repeated-signature box, sunrise, kite, and iterated-bubble cases covered by
  tests.  Terminal tadpoles are encoded as unit denominator-tree nodes, so pure
  CFF denominators remain `E`-surface-only.

Bounded pure CFF is constructed directly with E-surface denominator terms
rather than through a `CFF + (LTD - CFF)` correction.

### Uniform Repeated-Channel Sampling Scale

Repeated-channel interpolation normally uses the dimensionless variable
`q_C^0/OSE[C]`.  This is the default:

```bash
--uniform-numerator-sampling-scale none
```

For high repeated-channel numerator degrees, the interpolation can instead use
a runtime scale `M`, so inverse interpolation powers are `1/M^r` rather than
artificial `1/OSE[C]^r` factors:

```bash
python3 generalised_ltd.py build --family cff \
  --dot examples/graphs/box_pow3.dot \
  --energy-degree-bounds 3:4 \
  --uniform-numerator-sampling-scale beyond-quadratic \
  --pretty --show-details
```

Available policies are:

- `none`: default, use the existing `OSE[C]` normalization.
- `beyond-quadratic`: use `M` only for repeated-channel degree greater than 2.
- `all`: use `M` for every repeated-channel interpolation degree greater than 1.

Uniform JSONs record `graph.uniform_numerator_sampling_scale` and
`graph.uniform_scale_symbol="M"`. Energy maps can contain an integer `"m"`
coefficient, e.g. `{"i":[[3,1]],"x":[[0,-1]],"m":2,"c":"0"}` means
`OSE[3] - E[0] + 2*M`; variants can also carry `uniform_scale_power`, meaning
an extra factor `M^-power`.

Evaluate such JSONs with a non-zero scale:

```bash
python3 generalised_ltd.py evaluate \
  --orientation-json tmp/box_pow3_uniform_cff.json \
  --dot examples/graphs/box_pow3.dot \
  --numerator-expr auto \
  --uniform-scale -2.0
```

`test` can compare the default representation and uniform variants in one run:

```bash
python3 generalised_ltd.py test --dot examples/graphs/box_pow3.dot \
  --energy-degree-bounds 0:1,1:1,2:0,3:4 \
  --numerator-expr auto \
  --uniform-numerator-sampling-scale beyond-quadratic \
  --uniform-scales 1.0,-2.0,2.75
```

`M=0` is rejected. Real evaluators accept positive or negative real values.
Complex Symbolica evaluators accept complex `M`; real Symbolica evaluators
reject complex `M` cleanly.

The CFF builder does not treat repeated signatures as a special obstruction:
they are repeated denominator factors in the CFF expression.  For high-power
bounds, the builder first reduces a repeated channel as a whole, so a channel
such as the repeated `k2` sector in `sunrise_pow4.dot` with
`--energy-degree-bounds 2:5` keeps an E-surface-only CFF denominator instead of
falling into a lower-sector LTD correction.  For non-repeated high-power
contacts it decomposes multiloop lower denominators into loop-energy matroid
components and serializes terminal tadpoles as unit denominator-tree nodes.

### Reading bounded repeated-channel JSON

For a repeated channel `C` with representative edge `r`, multiplicity `nu`, and
total requested channel degree `d_C`, bounded pure CFF builds a Lagrange
interpolation in `q_C^0/OSE[r]`, rewrites every monomial in powers of
`D_C=(q_C^0)^2-OSE[r]^2`, and serializes each resulting term as a normal
orientation variant:

- `edge_q0` is the unique black-box numerator map for the orientation.
- `variants[*].meta.channel_reductions` records the derivation:
  `members`, `representative`, `power`, `degree_bound`, `sample`,
  `remaining_power`, `parity`, `cancelled_power`, `inverse_ose_power`, and the
  rational `coefficient`.
- `half_edges` contains the ordinary CFF half-edge factors plus
  `inverse_ose_power` copies of the representative channel edge.
- `num_surfaces` contains the cached numerator-only surface for the remaining
  factor `q_C^0` when `parity=1`; denominator surfaces remain `E`-only.

Representative CLI checks:

```bash
python3 generalised_ltd.py build --family cff \
  --dot examples/graphs/box_pow3.dot \
  --energy-degree-bounds 3:4 --pretty --show-details

python3 generalised_ltd.py build --family cff \
  --dot examples/graphs/sunrise_pow4.dot \
  --energy-degree-bounds 2:5 --pretty --show-details

python3 generalised_ltd.py test \
  --dot examples/graphs/sunrise_pow4.dot \
  --energy-degree-bounds 2:5 --numerator-expr auto
```

For `box_pow3.dot --energy-degree-bounds 3:4`, the repeated channel is
`[3,4,5]`, has power `3`, and uses interpolation nodes
`1,-1,0,2,-2`.  The first printed numerator map has five variants; for
sample `1` their channel metadata keeps denominator powers `1,2,2,3,3`,
with inverse OSE powers `4,2,3,0,1`.

For `sunrise_pow4.dot --energy-degree-bounds 2:5`, the repeated channel is
`[2,3,4,5]`, has power `4`, and uses nodes `1,-1,0,2,-2,3`.  The first
printed numerator map has six variants; for sample `1` their channel metadata
keeps denominator powers `2,2,3,3,4,4`, with inverse OSE powers
`4,5,2,3,0,1`.  This is the JSON form of the identity
`(q_C^0)^5/D_C^4 = OSE_C^4 q_C^0/D_C^4 + 2 OSE_C^2 q_C^0/D_C^3 + q_C^0/D_C^2`,
but implemented for a black-box numerator through interpolation samples.

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

The five-loop ultimate basis alignment test is still present but slow.  Run
it explicitly with:

```bash
HYBRID3D_RUN_SLOW=1 PYTHONPATH=. pytest -q tests/test_cli.py::test_ultimate_five_loop_bases_three_way_and_aligned_momenta
```

See `docs/generalised_ltd.pdf` for the derivation, JSON mapping, and bounded-degree
support structure.
