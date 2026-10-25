---
title: Secret providers in Ferry 2.0
product: ferry
version: 2.0
doc_type: guide
released: 2026-06-30
---

## Choosing a provider

`secrets.provider` in `ferry.yaml` selects where secrets come from: `file` (the encrypted
`ferry.secrets` of 1.5, the default), `vault` (a secrets server reached over HTTPS) or
`env` (variables of the machine that runs the deploy).

## The vault provider

Set `secrets.vault.address` and `secrets.vault.path`. Ferry authenticates with the token in
`FERRY_VAULT_TOKEN` and fetches secrets at deploy time; they are never written to the
repository.
