---
title: Progressive rollouts in Ferry 2.0
product: ferry
version: 2.0
doc_type: guide
released: 2026-06-30
---

## Rollout steps

`ferry deploy --progressive` deploys a release in steps. The default steps are 5, 25, 50
and 100 percent of the hosts; set `rollout.steps` in `ferry.yaml` to change them. Ferry
pauses 10 minutes between steps by default (`rollout.pause`).

## Automatic abort

During every pause Ferry reads the error rate and latency of the updated hosts. A rollout
aborts and rolls back on its own when the error rate goes above 2 percent or the p95
latency goes above 800 milliseconds. Both thresholds live under `rollout.abort`.

## Resuming and stopping

`ferry rollout status` shows the current step. `ferry rollout resume` skips the rest of a
pause; `ferry rollout stop` freezes the rollout where it is.
