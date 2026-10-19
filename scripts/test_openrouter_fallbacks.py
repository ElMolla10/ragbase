"""Three synthetic BYOK routing probes; never prints keys, prompts or answers.

Workspace prerequisite: Never use shared capacity for ANY model on both
providers. This script does not change workspace settings or create provider keys.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.openrouter_client import MODELS, PROVIDERS, sync_client, sync_completion  # noqa: E402


def main(output=None):
    from openai import APIConnectionError, APIStatusError

    results = []
    with sync_client() as client:
        cases = (
            ("standard", MODELS[0], PROVIDERS, MODELS),
            ("groq_only", MODELS[1], PROVIDERS[1:], (MODELS[1],)),
            # Keep Gemini primary while excluding its provider: OpenRouter must
            # select the fallback model within this same completion request.
            ("google_unavailable", MODELS[0], PROVIDERS[1:], (MODELS[1],)),
        )
        for label, primary, providers, expected_models in cases:
            try:
                response = sync_completion(
                    client,
                    model=primary,
                    providers=providers,
                    messages=[
                        {"role": "user", "content": "Reply with the single word OK."}
                    ],
                    max_tokens=512,
                )
                matched = response.model in expected_models
                provider = getattr(response, "provider", None)
                results.append(
                    {
                        "case": label,
                        "http_status": 200,
                        "model": response.model,
                        "reported_provider": provider
                        if isinstance(provider, str)
                        else None,
                        "requested_model": primary,
                        "allowed_providers": list(providers),
                        "passed": matched
                        and bool(
                            response.choices and response.choices[0].message.content
                        ),
                    }
                )
            except APIStatusError as exc:
                results.append(
                    {"case": label, "http_status": exc.status_code, "passed": False}
                )
            except APIConnectionError:
                results.append(
                    {
                        "case": label,
                        "http_status": None,
                        "error": "connection_unavailable",
                        "passed": False,
                    }
                )
            print(json.dumps(results[-1]), flush=True)
    report = {
        "synthetic_only": True,
        "shared_capacity_setting": "owner-reported-never-for-any-model",
        "cases": results,
        "passed": all(r["passed"] for r in results),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.output))
