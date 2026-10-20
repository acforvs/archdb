"""Check Git's publication file list; never print matched credential values."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    ".local",
    ".netlify",
    ".cache",
    ".generated",
    "site",
    "node_modules",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}
PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "Netlify token": re.compile(rb"\bnfp_[A-Za-z0-9_-]{20,}\b"),
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    "OpenAI token": re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b"),
    "AWS key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Slack token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "credential URL": re.compile(rb"https?://[^\s/:]+:[^\s/@]+@"),
    "credential assignment": re.compile(
        rb"""(?m)^\s*(?:export\s+)?[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY)[A-Z0-9_]*\s*=\s*["']?[A-Za-z0-9_./+-]{16,}"""
    ),
    "JSON credential": re.compile(
        rb"""(?i)["'](?:auth_token|access_token|api_key|client_secret|password|netlify_auth_token|github_token|refresh_token)["']\s*:\s*["'][^"'\s]{16,}["']"""
    ),
    "JWT": re.compile(rb"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"),
}


def check(root=ROOT):
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root).decode().split("\0")
    failures = []
    count = 0
    for entry in filter(None, entries):
        info, name = entry.split("\t", 1)
        mode, blob, stage = info.split()
        path = Path(name)
        if (
            FORBIDDEN.intersection(path.parts)
            or name.startswith(("data/raw/", "archdb/static/", "build/", "dist/"))
            or path.name.startswith(".env")
            or path.suffix.lower() in {".pem", ".key", ".pdf", ".sqlite"}
            or re.match(r"(?:credentials|secrets).*\.json$", path.name, re.IGNORECASE)
        ):
            failures.append((name, "excluded file"))
        if mode == "120000":
            failures.append((name, "symlink requires review"))
            continue
        if stage != "0":
            failures.append((name, "unresolved merge"))
            continue
        data = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=root)
        if len(data) > 5 * 1024 * 1024:
            failures.append((name, "file exceeds 5 MiB"))
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                failures.append((name, label))
        count += 1
    return count, failures


if __name__ == "__main__":
    count, failures = check()
    for path, reason in failures:
        print(f"{path}: {reason}")
    if failures:
        raise SystemExit(1)
    print(f"Checked {count} publication files: no excluded files or known credential patterns.")
