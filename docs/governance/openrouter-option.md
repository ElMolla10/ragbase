> Historical v3 implementation and live auto-router results below are superseded for current configuration by [the free Gemma update](gemma-json-mode.md).

# OpenRouter option — implementation record

18 October 2026. Constitution 3.0.0; execution plan 3.0. All seven phases remain.

The project owner requested optional hosted generation using `openrouter/auto`,
with no local model required. This explicitly amends the earlier local-only
restriction. The base development stack still disables generation by default.
Local Ollama remains a separate choice, with no automatic mode fallback.

Implementation includes a fixed OpenRouter gateway and a dedicated relay with a
file-mounted credential. The API retains its internal-only network and receives
no OpenRouter key. The relay receives no database credentials or document mounts
and publishes no port. It accepts only the fixed bounded text/quote request, adds
authentication and calls the approved HTTPS upstream. Provider errors are
normalized; redirects, arbitrary endpoints, tools and request extensions are
rejected. The relay's Docker edge network is not a host egress firewall.

Existing authorization, PostgreSQL RLS, immutable evidence, quote validation,
admission queues, revocation checks and cancellation remain active. Dynamic
response model/provider provenance travels in request-local events rather than
shared adapter state. Router names are not recorded as pinned checkpoints.
External processing is disclosed in streamed events, metadata and the terminal
client. The key-entry helper uses hidden input and writes only to ignored local
storage. See [setup](../openrouter.md).

Validation:

- 132 enterprise unit/contract/parser tests pass, including 29 hosted-mode tests.
  They use synthetic credentials and mocked HTTP; no model/provider is called.
- 55 actual PostgreSQL integration tests pass as the non-owner serving role,
  including two hosted-path cases proving denied access never reaches the gateway
  and forged quotations never become released answers.
- Ruff lint/format, policy assembly and Compose validation pass. The merged remote
  Compose topology includes no Ollama relay, no API edge access, no API key mount
  and no published remote-relay port. Key storage is excluded by Git and Docker.
- The disposable integration database/network were removed after testing. The initial implementation did not change the running development stack. The live follow-up below subsequently selected remote mode; installed Ollama/DeepSeek Harness settings were not changed.

These results establish development contracts only. At the initial implementation checkpoint, no real API key had been entered and no OpenRouter inference had run. That historical statement is superseded by the live follow-up below.
Health checks verify only credential configuration, not key validity or credits.
Live structured-output/privacy routing compatibility, remote tokenizer/context,
Arabic/company quality, target-server deployment, production egress enforcement,
backup recovery and company acceptance remain pending. Router selections may
change; no single remote checkpoint is qualified. Remote inference cannot pass an
offline inference gate. Local cancellation cannot guarantee remote processing or
billing stops after data has been received by a provider.


## Live setup follow-up

The owner privately saved the key and authorized continuation. The enterprise
stack now runs with the remote override; its former local Ollama relay was
removed while database/artifact volumes were retained. The key was not printed,
committed, copied into images or included in reports.

The fixed upstream accepted authenticated requests. Diagnostic synthetic calls
produced schema-valid responses, including a valid source quotation. The full
application smoke remains **failed**: four attempts released no generated answer,
and the subsequent completed response incorrectly reported insufficient evidence
for a retrieved sentence containing the requested number. These attempts are not
hidden or counted as passing quality cases. No company document was sent.

Live diagnostics also demonstrated that the auto router can select a mandatory
reasoning endpoint, which rejects disabling reasoning. Remote requests therefore
exclude reasoning from the response but allow it internally, and reserve 1,024
output tokens instead of the original 256. Local generation retains 256. The
4,096-unit development admission ceiling is unchanged; remote provider usage is
still checked and incomplete answers remain blocked. Safe fixed failure categories
are logged without exception text or provider bodies. This does not guarantee every
selected route fits the small allowance or returns a useful answer.

Connection and safe request handling work, but the quality smoke is not accepted.
Do not claim reliable company, Arabic or numerical performance from this setup.
The remaining production and model-quality gates above are still pending.
