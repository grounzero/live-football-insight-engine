# Security policy

## Scope

A portfolio project that replays public sample data locally. It has no authentication, no
persistence and no multi-tenancy, and is not intended for public deployment as-is.

## Reporting

Open a GitHub security advisory on this repository. Please do not open a public issue for a
suspected vulnerability.

## What is deliberate

- **No secrets.** No credentials, keys or tokens are needed or accepted. Configuration is a YAML
  file and `FI_`-prefixed environment variables.
- **Logs do not echo configuration or environment.** Structured logs carry only explicitly attached
  fields; exception records include the error type and message, never the environment.
- **Validation errors do not echo input.** The 422 handler returns the location and message only.
  FastAPI's default behaviour includes the rejected payload, which both leaks arbitrary client
  input back and breaks outright on values JSON cannot represent.
- **The service binds to `127.0.0.1` by default** and CORS allows only the local demo origin.
- **Model artifacts are unpickled** by the baseline loader. They are the project's own output, not
  untrusted input; do not point the registry at an artifact you did not produce.
- Dependencies are checked with `pip-audit` in CI.

## Data handling

The Metrica sample data is anonymised by its publisher and is never committed to this repository.
Nothing here attempts to de-anonymise it, and features are identity-invariant by design.
