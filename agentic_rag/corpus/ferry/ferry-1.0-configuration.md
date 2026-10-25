---
title: Ferry 1.0 configuration reference
product: ferry
version: 1.0
doc_type: reference
released: 2024-06-01
---

## The ferry.toml file

`ferry.toml` has three tables: `[service]` (name, build command), `[hosts]` (addresses and
SSH user) and `[deploy]` (health checks, hooks and release retention).

## Hooks

`pre_deploy` runs on each host before the symlink switches; `post_deploy` runs after the
restart. A hook that exits with a non-zero status aborts the deploy on that host.

## Release retention

Ferry 1.0 keeps the last 5 releases on every host and deletes older ones after a
successful deploy. The number is fixed in this release.

## Environment variables

Ferry 1.0 loads environment variables for the service from a plain `.env` file next to
`ferry.toml` and uploads it with every release. The file is not encrypted.
