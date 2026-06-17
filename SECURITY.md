# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅ |

## Reporting a Vulnerability

Open an issue at https://github.com/Matrix-Research-Ai/tabula-rasa/issues
with the label `security`.

## Known Security Considerations

### Code Sandbox (`code_sandbox.py`)

The `code_sandbox.py` module executes arbitrary Python code via
`exec()` / `subprocess`. This is **not sandboxed** — there is no
`bubblewrap`, `nsjail`, or `seccomp` wrapper. Do not expose the
code sandbox endpoint to untrusted users or the public internet.

### API Servers

The HTTP servers on ports 8000/8002 have **no authentication,
no CORS restrictions, and no rate limiting**. They are designed
for local development only. Running them on a public network
exposes your model and system to arbitrary queries.

### Model Weights

Trained model weights (`.pt` files) may contain sensitive learned
information. Use `pii_scrubber.py` to redact PII from training data
when training on user-provided content.

## Recommended Hardening

1. **Do not expose API servers to the internet** without adding
   API-key middleware and rate limiting.
2. **Wrap `code_sandbox.py`** in `bubblewrap`/`nsjail` with
   seccomp filters, CPU/memory timeouts, and no network access
   before any multi-user deployment.
3. **Keep dependencies updated** — run `dependabot` monthly
   (`.github/dependabot.yml` is configured).
4. **Use `pii_scrubber`** to detect and redact emails, phone
   numbers, SSNs, and credit card numbers from training data.
