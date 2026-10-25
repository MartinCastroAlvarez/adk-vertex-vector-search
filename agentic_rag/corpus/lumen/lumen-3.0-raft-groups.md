---
title: Raft groups in Lumen 3.0
product: lumen
version: 3.0
doc_type: guide
released: 2026-02-10
---

## How Raft groups work

A Raft group is a set of three or five Lumen servers that elect a leader. The leader
accepts writes and acknowledges one only after a majority of the group has stored it, so
an acknowledged write survives the loss of a minority of servers.

## Creating a group

Run `lumen group create --members db1:7410,db2:7410,db3:7410` on any member. Groups of
three tolerate one failed server; groups of five tolerate two. Even sizes are refused.

## Election timeout and heartbeats

The leader sends heartbeats every `raft.heartbeat_ms`, 150 milliseconds by default. A
follower that hears nothing for `raft.election_timeout_ms`, which defaults to 1500
milliseconds, starts an election. Keep the election timeout at least ten times the
heartbeat interval.
