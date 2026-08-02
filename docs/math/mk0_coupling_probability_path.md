# MK0 coupling and probability-path specification

## Coupling is latent and algorithmic

For each training pair `(x_src,x_target)`, create aligned sequences `z_src,z_target` over `{A,C,G,U,EPSILON}`. Removing EPSILON must recover the two inputs exactly. The primary construction is unit-cost Levenshtein optimal alignment with the frozen deterministic tie-break:

```text
MATCH, then DEL, then INS, then SUB, then lexical nucleotide order
```

This order is an algorithmic convention, not a biological statement. Every record validates `schemas/coupling_manifest_v1.schema.json` and sets:

```yaml
path_is_observed: false
path_semantics: latent_algorithmic
```

The primary result is accompanied by two development-only sensitivities: sampling from the set of optimal alignments and changing equivalent edit order. Their results are never selected using final labels.

## Frozen joint path

For aligned coordinate `i`, with monotone schedule `kappa(t)`:

\[
p_t(z_i)=(1-\kappa(t))\delta_{z_i^{src}}+\kappa(t)\delta_{z_i^{tar}}.
\]

MK0-v1 freezes the joint distribution, not only its marginals:

\[
p_t(z\mid z^{src},z^{tar})=\prod_i p_t(z_i\mid z_i^{src},z_i^{tar}).
\]

For every changing coordinate `i`, draw an independent switch clock with `P(tau_i <= t)=kappa(t)`. Given a uniform draw `u_i`, cubic uses `tau_i=u_i^(1/3)` and linear uses `tau_i=u_i`. Unchanged coordinates do not receive target-switch events.

The remaining switch set is

\[
R_t=\{i:z_i^{src}\ne z_i^{tar},\ \tau_i>t\}.
\]

With a shared schedule, every remaining coordinate has conditional target hazard

\[
\rho(t)=\frac{\dot\kappa(t)}{1-\kappa(t)}.
\]

For cubic, `rho(t)=3t^2/(1-t^3)`; for linear sensitivity, `rho(t)=1/(1-t)`. Rates are never evaluated exactly at `t=1`; the frozen `time_eps` and clipping counters come from `math_kernel_v1.yaml`. A future coordinate-specific schedule requires a new derivation and kernel hash.

## From augmented path to observable transitions

Removing EPSILON from `z_t` produces the current observable sequence. Each remaining aligned switch maps to an observable atomic action when legal. Several auxiliary coordinates can map to one canonical action, and several actions can map to one next observable string. The implementation therefore performs two distinct audits:

1. preserve the full runtime mapping when determining whether next **extended** states are equal; and
2. aggregate all target weights and all model rates that truly reach the same next extended state before taking a logarithm.

Repeated symbols such as `AA -> A`, `A -> AA`, `AAA -> AA` and `AA -> AAA` are mandatory oracle cases. Observable-string equality alone is insufficient: source token IDs, inserted-event IDs, gap IDs, history and budget may distinguish extended states. Conversely, auxiliary multiplicity that maps to the same canonical transition must be summed rather than double-logged.

## Runtime/auxiliary firewall

`Z_aux=(z_src,z_t,z_target)` is permitted only inside coupling and target-weight construction. It is prohibited from:

- runtime state serialization;
- source/current encoder inputs;
- adapter, operation/token rate head or STOP head inputs;
- edit-history or source-current mapping features; and
- sampler decisions.

The leakage test holds every inference-visible field and model parameter fixed, then permutes or replaces the paired target/alignment. The complete output rate vector must remain equal under the frozen tolerance. Changing `Z_aux` may change target transition weights and loss labels, but not predicted rates.

## Rejected-path semantics

If an auxiliary target action conflicts with UTR grammar, protected anchors, length or budget constraints, the record is not silently shortened. It is either:

- rejected with a reason and denominator contribution; or
- repaired by a declared coupling procedure, revalidated, and assigned a new alignment hash.

Both dispositions are append-only records in the coupling artifact. The acceptance report includes the rejected fraction and reconstruction denominator.

## Required oracles

The CPU oracle verifies exact source/target reconstruction, optimal cost, deterministic canonical tie-breaking, sampled-optimal support, product-path probabilities, independent-clock empirical checks, schedule derivatives, remaining-switch membership, `rho(t)`, target-weight aggregation and the target-alignment firewall. These are E0 tests; they do not show that any latent alignment is biologically true.
