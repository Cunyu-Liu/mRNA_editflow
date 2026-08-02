# MK0 transition-level Bregman/rate objective

## Generator and transition aggregation

At a fixed external time `t`, let legal atomic action rate be `u_theta(a|S_t) >= 0`. The action factorization is:

\[
u(INS(g,v))=m_{g,v}^{ins}\lambda_g^{ins}Q_g^{ins}(v),
\]

\[
u(SUB(i,v))=m_{i,v}^{sub}\lambda_i^{sub}Q_i^{sub}(v),\quad
u(DEL(i))=m_i^{del}\lambda_i^{del},\quad
u(STOP)=m^{stop}\lambda^{stop}.
\]

Hard masks are applied before legal-token normalization. Operation rates are nonnegative but do not sum to one. Only the action distribution conditioned on an event is normalized.

For a next extended state `S'`:

\[
U_\theta(S'\mid S_t)=\sum_{a:T_Y(Y_t,a)=Y'}u_\theta(a\mid S_t).
\]

The off-diagonal generator entry is this aggregated rate, while the diagonal is minus total legal hazard. Enumerating a finite state must give generator row sum zero under the frozen float64 tolerance.

## Target transition weights

For remaining auxiliary switches `R_t`, map each switch to its legal observable action and then to the next extended state. With the frozen common schedule:

\[
W_t(S'\mid S_t,Z_{aux})=
\sum_{i\in R_t:T_Y(Y_t,\bar a(e_i))=Y'}\rho(t).
\]

If a target action is illegal, the sample is rejected or coupling-repaired and rehashed. Setting its target weight to zero without a ledger entry is a failure.

## Objective

Ignoring constants independent of model parameters, the edit term is:

\[
\mathcal L_{EF}=\sum_{S'\in\mathcal N_{edit}}U_\theta(S'\mid S_t)
-\sum_{S':W_t(S')>0}W_t(S')\log U_\theta(S'\mid S_t).
\]

For one transition, `ell(U,W)=U-W log U` and

\[
\frac{\partial \ell}{\partial U}=1-\frac{W}{U}.
\]

When multiple actions share a next extended state, every action receives the same transition-level multiplier through `dU/du_a=1`. Replacing the term with `sum_a W_a log u_a` is generally wrong. Action-level decomposition is allowed only when a formal one-to-one argument proves equivalence for the serialized extended state.

## Repeated-symbol oracle

The brute-force oracle includes repeated insertion/deletion and equivalent-script cases. For each case it:

1. enumerates legal actions;
2. applies the deterministic extended-state update;
3. groups by canonical next-state serialization and hash;
4. independently sums model action rates and auxiliary target multiplicities;
5. evaluates the grouped expression directly;
6. compares the implementation value; and
7. compares automatic gradients with an independently evaluated analytic or central finite-difference reference.

The report preserves observable collisions that remain distinct in extended state due to source origin, gap identity or history. It also preserves true extended-state collisions caused by auxiliary multiplicity. This prevents both under-aggregation and false aggregation.

## Numerical and semantic guards

- A logarithm is evaluated only where aggregated target weight is positive and aggregated model transition rate is finite and strictly positive.
- Nonfinite loss, rate or gradient yields `FAILED_WITH_EVIDENCE`; no silent epsilon repair is used to convert it into a PASS.
- Float64 CPU oracle tolerances and finite-difference tolerances are frozen separately in `math_kernel_v1.yaml`.
- HALTED states contribute no edit-flow term.
- STOP uses the separately derived survival loss; it is not inserted as an unproved edit target.
- `L_MK0 = L_EF + alpha_stop L_stop`, with `alpha_cond=alpha_reg=0`.
- Critic reward, final evaluator values and undefined regularizers are absent.

The held-out flow objective is not labelled an exact normalized sequence likelihood or exact NLL.
