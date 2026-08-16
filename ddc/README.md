# DDC pre-hardware assurance

This directory records the DDC-style assurance boundary for WebGazer AAC v2 before browser/webcam/AAC-board testing.

## What the pass added

The assurance model separates:

- observation from action authority;
- predicted gaze from ground truth;
- tracking quality from target confidence;
- calibration evidence from weaker adaptive/inferred evidence;
- runtime behavior from persisted calibration state;
- validated artifact identity from intended source changes.

The machine-readable transition contract is `pre-hardware-assurance.json`. The DDC Product Adapter declaration is `product-adapter.json`. The bounded Work Order and observed local validation record are preserved separately.

## Findings

The DDC pass found pre-hardware gaps not covered by the original v2 test set:

1. WebGazer interception needed an explicit reversible teardown boundary (`uninstall`).
2. Full calibration clearing needed to remove persistent calibration, in-memory calibration/model evidence, evidence lineage, and transient eye evidence together.
3. Raw-normalized training evidence and provenance needed bounded retention and preservation across PCA feature-space transitions.
4. Display DPR belongs to the calibration validation boundary.
5. Artifact identity itself must be a release invariant: validation for one byte-identical candidate cannot silently transfer to another artifact.

A local hardened candidate implementing these changes passed 80/80 deterministic pre-hardware checks, including a 20,000-frame bounded-session simulation. That result is recorded in `PRE-HARDWARE-VALIDATION.json` and is bound to its SHA-256.

## Current promotion state

The locally verified hardened candidate is **not yet the repository root runtime**. An attempted connector transfer produced a different Git blob and was rejected before attachment to the branch. `PROMOTION-GATE.json` records that event and the exact identities required for promotion.

This is intentional fail-closed behavior. The draft PR must not be treated as hardware-test-ready until the repository-promoted `webgazer-aac.js` reproduces the verified candidate identity and the 80-check gate is rerun against that promoted artifact.

## Non-claims

This evidence does not establish clinical validation, medical-device status, real-world accuracy improvement, or human AAC effectiveness. It also does not claim that DDC Platform providers executed: the current record is a local deterministic reproduction using DDC contracts and invariants, not a DDC Router/provider Run.
