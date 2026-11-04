"""Cache primary sources and UCI data; pin URLs and SHA-256 in a manifest.

Run once with network access: python -m research.fetch_stage1
Later runs verify the existing cache, without resolving moving upstream refs.
Remove the manifest explicitly to start a new audit snapshot.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/sources.json"


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "HDDA-stage1-research"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main():
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        for item in manifest["artifacts"]:
            path = ROOT / item["path"]
            if not path.exists() and not item.get("rehydrate", True):
                print(f"Optional volatile snapshot unavailable: {path.name}")
                continue
            content = path.read_bytes() if path.exists() else download(item["url"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(f"Verified {len(manifest['artifacts'])} pinned artifacts")
        return

    artifacts = []
    revisions = {}

    def save(url, relative, rehydrate=True):
        content = download(url)
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append({"url": url, "path": relative, "bytes": len(content), "rehydrate": rehydrate,
                          "sha256": hashlib.sha256(content).hexdigest()})
        print(relative, len(content))
        return content

    repositories = {
        "statsmodels/statsmodels": ("main", ["statsmodels/multivariate/cancorr.py",
                                            "statsmodels/multivariate/tests/test_cancorr.py",
                                            "CONTRIBUTING.md", "AGENTS.md"]),
        "cran/CCA": ("master", ["DESCRIPTION", "NAMESPACE", "R/rcc.R", "R/comput.R",
                                  "man/nutrimouse.Rd"]),
        "andersonwinkler/PermCCA": ("master", ["permcca.m", "README.md"]),
    }
    for repo, (ref, paths) in repositories.items():
        commit = json.loads(download(f"https://api.github.com/repos/{repo}/commits/{ref}"))["sha"]
        revisions[repo] = commit
        for path in paths:
            save(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}",
                 f"research/cache/{repo.split('/')[-1]}/{path}")
    save("https://archive.ics.uci.edu/static/public/72/multiple+features.zip",
         "data/raw/multiple_features.zip")
    save("https://archive.ics.uci.edu/dataset/72/multiple+features",
         "research/cache/uci_multiple_features.html", rehydrate=False)
    for package in ["statsmodels", "cca-zoo", "scikit-learn"]:
        save(f"https://pypi.org/pypi/{package}/json", f"research/cache/{package}_pypi.json", rehydrate=False)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"audit_date": "2026-11-04", "revisions": revisions,
                                    "artifacts": artifacts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
