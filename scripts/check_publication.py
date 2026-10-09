"""Check publishable source files and optionally export them without Git history."""

import argparse
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRS = {".git", ".idea", ".vscode", ".venv", "venv", "env", "out",
                "data", "private", "__pycache__", ".ipynb_checkpoints"}
PRIVATE_SUFFIXES = {".dta", ".sas7bdat", ".sas7bcat", ".csv", ".tsv", ".parquet",
                    ".feather", ".h5", ".hdf5", ".sqlite", ".sqlite3", ".db",
                    ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".pt", ".pth",
                    ".ckpt", ".pem", ".key", ".p12", ".pfx", ".pyc"}
PATTERNS = {
    "personal filesystem path": re.compile(r"/(?:Users|home)/[\w.-]+/|[A-Za-z]:\\+Users\\+", re.I),
    "email address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "credential token": re.compile(r"AKIA[0-9A-Z]{16}|(?:ghp_|github_pat_|sk-proj-)[\w-]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "literal credential assignment": re.compile(
        r"\b(?:api_key|password|passwd|secret|access_token|auth_token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]", re.I),
    "authenticated URL": re.compile(r"https?://[^\s/\"']+:[^\s/\"']+@"),
}


def source_files():
    """Include tracked and new non-ignored files, including tracked ignored files."""
    if (ROOT / ".git").exists():
        result = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
        names = sorted(set(result.decode().split("\0")) - {""})
        return [ROOT / name for name in names if (ROOT / name).exists()]
    # Also works in an extracted archive without Git.
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and not set(p.relative_to(ROOT).parts) & PRIVATE_DIRS)


def check(files):
    issues = []
    for path in files:
        name = path.relative_to(ROOT)
        if path.suffix.lower() == ".ipynb":
            issues.append((name, "notebook excluded from public release"))
            continue
        if path.is_symlink():
            issues.append((name, "symlink requires manual review"))
            continue
        if (set(name.parts) & PRIVATE_DIRS or path.suffix.lower() in PRIVATE_SUFFIXES
                or path.name == ".DS_Store" or path.name.startswith(".env")
                or path.name.lower().startswith(("credentials", "secrets"))):
            issues.append((name, "private/local artifact included in source"))
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append((name, "binary file requires manual review"))
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                issues.append((name, label))
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Write out/FBML-public-source.zip")
    args = parser.parse_args()
    files = source_files()
    issues = check(files)
    for name, reason in issues:
        print(f"FAIL {name}: {reason}")  # Never print detected secret values.
    if issues:
        raise SystemExit(1)
    print(f"PASS: checked {len(files)} source files. Git history is not covered.")
    if args.export:
        destination = ROOT / "out" / "FBML-public-source.zip"
        destination.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, str(Path("FBML-public-source") / path.relative_to(ROOT)))
        print(f"Exported {destination.relative_to(ROOT)} without Git history or ignored files.")


if __name__ == "__main__":
    main()
