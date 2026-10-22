# DDC Radial / Frequency cross-repository eye-tracking audit — 2026-10-22

## Scope and identity
- `altrudev/webgazer-aac` main `c264cdfcac1fd08c3c5355e96edc453b74fd5530`
- `altrudev/JelloOS` main `a17ba3aa1bf45027e9ce1d9c589dd612c9055150`
- `altrudev/Frequency` main `1071cf1` (snapshot used to inspect `skills/ddc-radial-review/SKILL.md`, `docs/FREQUENCY-AUDITOR-v0.2.md`)
- Deployed `https://altru.dev/assets/labs/hybrid-live.html` and referenced JavaScript.
- Read-only technical review: **not an authorization, safety certification, or release permit**.

## Executed checks
- `webgazer-aac: npm test`: **19/19 pass**.
- `webgazer-aac: npm run test:iris`: **2/2 pass**.
- `node --check lab/hybrid-live.js`: pass.
- Deployed `hybrid-live.js` hash = repo original: SHA-256 `12af709e45bace2e0a61b4d2076e3c31a3fad5d9c60875a050a3583c33fe92ad`.
- Hosted lab HTML returned HTTPS 200.
- Previous simulated-camera Chromium smoke test loaded the lab and reached 'Measuring'; synthetic feed provided no face. **No physical webcam acceptance**.
- Previous user diagnostic upstream-only: gaze target errors 583px, 274px; mean 429px. n=2, not a full error distribution.
- Frequency Auditor Rust tests started in VPS; report PASS only if process exit is inspected. Do not infer a result from a running build.

## DDC Radial findings

| ID | Severity | Evidence | Contradiction / risk | Falsification test | Downstream requirement |
|---|---|---|---|---|---|
| EYE-001 | Critical | Baseline errors 583px/274px; no holdout validation | Prediction callbacks wrongly conflated with accurate fixation | 9–16 withheld targets, zone-wise median/P90, repeat head-motion conditions | No gaze-driven AAC action |
| EYE-002 | High | `hybrid/observation-bridge.js`: pair by time, `qualifiedForDwell:false` | Co-temporal predictions aren't from exactly same decoded video frame | Instrument `requestVideoFrameCallback` frame ID/mediaTime in both models | Frame provenance and synchronization |
| EYE-003 | High | `hybrid/head-frame-evidence.js` uses eye baseline and nose proxies | 2D proxy not six-degree-of-freedom head pose nor anatomical eye angle | Calibrated pose matrix ground truth and head-only motions | Label units honestly; no gaze compensation claim |
| EYE-004 | High | JelloOS `contexts/GazeContext.tsx` calls clmtrackr; `index.html` calls TFFaceMesh | Multiple tracker/runtime behaviors | One version-pinned runtime contract and equivalent acceptance tests | One shared engine adapter |
| EYE-005 | High | JelloOS `saveDataAcrossSessions(true)` and `enableAdaptiveRecalibration()` | Persistent or self-training data contaminates independent testing | Disable persistence; hash/trust boundary tests; holdout frozen | Explicit consent, local retention controls |
| EYE-006 | High | JelloOS eye-patch diagnostic threshold `l < 40` | Dark-region fraction is not iris/pupil localization | Annotated iris contour ground-truth images | Replace proxy with per-eye ellipse/landmark residual |
| EYE-007 | Medium | External WebGazer 3.5.3 and MediaPipe CDNs/model URLs in lab | Network and distribution dependence; pinned URLs alone aren't asset integrity | Offline rebuild/cache, integrity manifests and license inventory | Versioned asset manifest, permitted model distribution |
| EYE-008 | High | README LGPL-by-valuation claim in JelloOS vs LICENSE GPLv3 | Upstream rights cannot be modified by an unsupported README clause | Review exact contributor and upstream licenses | Remove/reconcile unverified relicensing claim |
| EYE-009 | High | Production lab and JelloOS remain separate despite common iris library | Merged source does not imply deployed application integration | Browser event-to-AAC wiring test, no-activation negative | Explicit deploy boundary/acceptance |
| EYE-010 | Medium | Upstream only diagnostic has 93% valid callbacks but 429px mean error | Tracking availability misrepresented as gaze accuracy | Separate availability, spatial error, delay and stability dashboards | Accuracy gate must be independent of callbacks |

## Frequency interpretation
Map to DDC Radial identity/revision, independent observation, failure domain, provenance, information flow, TOCTOU, recovery and self-protection obligations. A green JS unit suite does not confer execution/release authority. Frequency's read-only Auditor v0.2 does not itself mutate the target repo or certify AAC performance. Any release needs separate verified authorization and externally observed webcam acceptance.

## Recommended repair order
1. G0: Tag and freeze tested dependency/manifests and licenses; unify JelloOS runtime and remove unverified biometric quality claims.
2. G1: Validate iris centers against image-ground-truth with visible per-eye crops and real camera, including mirror and occlusion.
3. G2: Add MediaPipe 3D head-pose matrices and explicit camera/display coordinate conventions. Replace 2D-only proxy labels.
4. G3: Frame-level source ID and timestamp linkage; reject stale or unmatched evidence.
5. G4: Compare calibrated WebEyeTrack and WebGazer against held-out targets; measure drift and failure across lighting, head movement and users.
6. G5: Permit only validated gaze actions; accessible keyboard, mouse, switch and touch alternatives remain available.

**Audit verdict: FAIL for gaze-controlled AAC release; PASS for a contained diagnostic prototype with unresolved runtime/hardware accuracy gates.**
