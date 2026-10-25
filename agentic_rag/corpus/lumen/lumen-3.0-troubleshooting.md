---
title: Troubleshooting Lumen 3.0
product: lumen
version: 3.0
doc_type: troubleshooting
released: 2026-02-10
---

## LUM-311: quorum lost

LUM-311 means the Raft group cannot reach a majority, so it refuses writes. Bring failed
members back or, as a last resort, shrink the group with `lumen group force-reconfigure`.

## Servers refuse to start after upgrading

A 3.0 server refuses to start when its configuration still contains `mirror.*` or
`auth_password` settings. Remove them and convert the replicas into a Raft group.

## Frequent leader elections

Frequent elections usually mean the network drops heartbeats. Raise
`raft.election_timeout_ms` above its 1500 millisecond default before raising the
heartbeat interval.
