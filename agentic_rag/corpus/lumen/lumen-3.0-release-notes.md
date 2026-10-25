---
title: Lumen 3.0 release notes
product: lumen
version: 3.0
doc_type: release-notes
released: 2026-02-10
---

## Highlights

Lumen 3.0 replaces Mirror replication with Raft groups, which replicate every write to a
majority of servers before acknowledging it. It adds mutual TLS, tiered compaction and
incremental snapshots, and raises the maximum value size to 16 MiB.

## Breaking changes

Mirror is removed: servers that still have `mirror.*` settings refuse to start. The
`auth_password` setting, deprecated in 2.0, is removed; use API tokens or mutual TLS. The
default port stays 7410.

## Upgrading from 2.0

Upgrade replicas first, then the primary, then convert the set into a Raft group with
`lumen group create --from-mirror`. Run `lumen migrate` on every server; 1.x data
directories must be migrated to 2.0 first.
