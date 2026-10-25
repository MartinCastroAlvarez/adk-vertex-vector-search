---
title: Snapshots and backups in Lumen 3.0
product: lumen
version: 3.0
doc_type: guide
released: 2026-02-10
---

## Incremental snapshots

`lumen snapshot create --incremental` stores only the segments that changed since the
previous snapshot. The first snapshot of a server is always full. Restore with
`lumen snapshot restore <id>`, which replays the chain of incremental snapshots.

## Snapshot retention

Lumen keeps the last 7 snapshots by default; set `snapshot.retention` to change it. When a
full snapshot is deleted, the incremental snapshots that depend on it are merged first.

## Backups in a Raft group

Take snapshots on a follower. A snapshot records the Raft index it covers, so restoring a
single member lets it catch up from the leader.
