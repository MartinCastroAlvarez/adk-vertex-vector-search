---
title: Getting started with Ferry 1.0
product: ferry
version: 1.0
doc_type: guide
released: 2024-06-01
---

Ferry is a command-line tool that deploys web services to a fleet of servers over SSH.

## Creating a project

Run `ferry init` in your repository. It writes `ferry.toml` with a service name, the list
of hosts and the build command. Ferry 1.0 reads only `ferry.toml`.

## Your first deploy

`ferry deploy` builds the service, uploads the release to every host, switches the
`current` symlink and restarts the service. A deploy fails if any host does not pass its
health check.

## Health checks

After a restart Ferry polls the `health_path` of each host, `/healthz` by default. A host
that does not answer with HTTP 200 within the health check timeout of 30 seconds fails the
deploy.
