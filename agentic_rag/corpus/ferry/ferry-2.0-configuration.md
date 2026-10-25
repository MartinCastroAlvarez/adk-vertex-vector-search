---
title: Ferry 2.0 configuration reference
product: ferry
version: 2.0
doc_type: reference
released: 2026-06-30
---

## The ferry.yaml file

`ferry.yaml` has four sections: `service`, `hosts`, `deploy` and `rollout`. Run
`ferry config validate` to check it.

## Hooks

`pre_deploy` and `post_deploy` work as in 1.x. The new `on_failure` hook runs on a host
when its deploy fails or a rollout aborts, for example to page the on-call engineer.

## Health checks

`deploy.health_path` defaults to `/healthz` and `deploy.health_timeout` to 60 seconds.

## Release retention

`deploy.keep_releases` defaults to 10.
