---
title: Secrets in Ferry 1.5
product: ferry
version: 1.5
doc_type: guide
released: 2025-05-15
---

## Storing a secret

`ferry secrets set DATABASE_URL` reads the value from standard input and stores it
encrypted in `ferry.secrets` in the repository. Values are encrypted with an age key.

## The encryption key

Ferry looks for the age key in `~/.config/ferry/age.key`. Create one with
`ferry secrets keygen`. Anyone who deploys needs the key; never commit it.

## Secrets at deploy time

During a deploy Ferry decrypts the secrets locally and sends them to each host over the
SSH connection; they are written to the release directory with mode 0600.
