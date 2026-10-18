"""Create local-only development credentials without printing them or overwriting."""

import json
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
VIEWER = "33333333-3333-4333-8333-333333333333"
WORKSPACE_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
WORKSPACE_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def main():
    target = ROOT / ".env.enterprise"
    if target.exists():
        raise SystemExit(
            "Development configuration already exists; leaving it unchanged"
        )
    owner_password, app_password, admin_password = [
        secrets.token_urlsafe(32) for _ in range(3)
    ]
    tokens = {label: secrets.token_urlsafe(32) for label in ("alice", "bob", "viewer")}
    identities = {tokens["alice"]: ALICE, tokens["bob"]: BOB, tokens["viewer"]: VIEWER}
    settings = {
        "DEV_POSTGRES_PASSWORD": admin_password,
        "DEV_OWNER_PASSWORD": owner_password,
        "DEV_APP_PASSWORD": app_password,
        "ENTERPRISE_DATABASE_URL": f"postgresql+asyncpg://ragbase_app:{app_password}@database:5432/ragbase_enterprise",
        "ENTERPRISE_MIGRATION_DATABASE_URL": f"postgresql+asyncpg://ragbase_owner:{owner_password}@database:5432/ragbase_enterprise",
        "ENTERPRISE_DEV_TOKENS": json.dumps(identities, separators=(",", ":")),
    }
    local = ROOT / ".local"
    local.mkdir(exist_ok=True)
    with target.open("x", encoding="utf-8") as file:
        file.write(
            "# Generated local development secrets. Never commit or deploy to production.\n"
        )
        for key, value in settings.items():
            file.write(f"{key}='{value}'\n")
    with (local / "dev-access.json").open("x", encoding="utf-8") as file:
        json.dump(
            {
                "development_only": True,
                "tokens": tokens,
                "workspace_a": WORKSPACE_A,
                "workspace_b": WORKSPACE_B,
                "identities": {"alice": ALICE, "bob": BOB, "viewer": VIEWER},
            },
            file,
            indent=2,
        )
    print(
        "Created ignored .env.enterprise and .local/dev-access.json; credentials were not printed."
    )


if __name__ == "__main__":
    main()
