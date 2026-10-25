---
title: Ferry 1.5 release notes
product: ferry
version: 1.5
doc_type: release-notes
released: 2025-05-15
---

## Highlights

Ferry 1.5 adds canary deploys, encrypted secrets with `ferry secrets`, and configurable
release retention.

## Changes

`keep_releases` in the `[deploy]` table now sets how many releases stay on each host; it
defaults to 10 (Ferry 1.0 always kept 5). Plain `.env` files still work but print a
warning suggesting `ferry secrets`.
