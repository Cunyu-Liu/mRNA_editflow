# B0-E2 resource authorization

## Authorization

On 2026-07-30, the project owner explicitly authorized an unrestricted
increase in B0 compute-resource budget so the project can continue.

## Scope of this E2 diagnostic

This authorization permits a fresh, isolated exact-capacity diagnostic with
larger compute, disk-spill, and wall-clock safety envelopes.  It does **not**
alter the scientific B0 acceptance criteria, the frozen B0 limits, data
selection, split semantics, or any evidence already retained by E1.

## Physical safety envelope

The configuration permits at most 5 TB of diagnostic spill while preserving
8 TB of free `/mnt` space, uses at most 64 GiB RSS, and stops after 14 days.
It runs at low CPU and I/O priority and records a fresh run ID, manifest,
event log, system metrics, and terminal evidence.  Reaching any physical
safety limit is a new `SAFE_RESOURCE_PAUSE`, never an approximation or a B0
pass claim.

## Provenance

Parent E1 diagnostic: `B0_capacity_20260730T043737Z_f8e30a4`.

E2 config: `configs/b0_capacity_diagnostic_e2_resource_authorized_v1.json`.
