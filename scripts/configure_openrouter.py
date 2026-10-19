"""Enter/rotate the OpenRouter credential privately; never prints its value."""

from getpass import getpass
from pathlib import Path


def main():
    key = getpass("OpenRouter API key (hidden input): ").strip()
    if not 20 <= len(key) <= 4096 or not key.isascii() or any(c.isspace() for c in key):
        raise SystemExit("Invalid key format; no file was changed.")
    target = Path(__file__).resolve().parents[1] / ".local/openrouter-api-key"
    target.parent.mkdir(exist_ok=True)
    target.write_text(key, encoding="utf-8")
    target.chmod(0o600)
    print(
        "Saved the private local credential. It is excluded from Git and Docker builds."
    )


if __name__ == "__main__":
    main()
