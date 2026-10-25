---
title: Getting started with Lumen 1.4
product: lumen
version: 1.4
doc_type: guide
released: 2024-03-12
---

Lumen is an embedded key-value database with an optional network server. This guide
installs Lumen 1.4, starts a server and stores the first keys.

## Installing Lumen

Lumen 1.4 ships as a single binary for Linux and macOS. Download the archive for your
platform, unpack it and put the `lumen` binary on your PATH. Run `lumen version` to check
the installation; it prints the version and the build date.

## Starting the server

Start a server with `lumen serve --data ./data`. The data directory is created if it does
not exist and holds the write-ahead log and the segment files. By default the server
listens on port 7400 on all interfaces. Use `--bind 127.0.0.1` to accept local connections
only.

## Reading and writing keys

Use `lumen put greeting hello` to store a value and `lumen get greeting` to read it back.
Keys are byte strings of up to 512 bytes. Values can be up to 1 MiB in Lumen 1.4. Use
`lumen delete greeting` to remove a key; deleted keys are reclaimed by compaction.
