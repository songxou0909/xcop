"""Prepare an xcop release; add --publish to commit and upload it to GitHub.

This edits managed scripts in place. Run it in the publication folder, not a
configured cluster installation. Publishing uses GitHub's latest main as its
base and a separate Git index, leaving the local branch and staging area alone.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from xcop_setup import (
    SetupWizard,
    UPDATE_BRANCH,
    UPDATE_MANIFEST_PATH,
    UPDATE_REPOSITORY,
    UPDATE_REPOSITORY_URL,
    UpdateError,
    generate_update_manifest,
    update_script_sha256,
)


PUBLISH_ORIGIN_URLS = {
    f"https://github.com/{UPDATE_REPOSITORY}",
    f"https://github.com/{UPDATE_REPOSITORY}.git",
    f"git@github.com:{UPDATE_REPOSITORY}.git",
    f"ssh://git@github.com/{UPDATE_REPOSITORY}.git",
}


class PublishError(Exception):
    """An update could not be committed or uploaded."""


def git(root, *args, data=None, env=None):
    """Run Git with literal arguments and preserve binary file contents."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, input=data, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, timeout=300,
        )
    except FileNotFoundError as exc:
        raise PublishError("Git is not installed or is not on PATH. Install Git and retry.") from exc
    except subprocess.TimeoutExpired as exc:
        raise PublishError(
            f"Git {args[0]} timed out. Check your connection and GitHub sign-in, then retry."
        ) from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise PublishError(f"Git {args[0]} failed:\n{detail}")
    return result.stdout


def check_publish_repository(root):
    """Check the destination and author before cleaning publication files."""
    top = git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(top).resolve() != root.resolve():
        raise PublishError("Run this script from an xcop Git checkout, not a subfolder or ZIP copy.")
    for options in ((), ("--push",)):
        urls = git(root, "remote", "get-url", *options, "--all", "origin").decode("utf-8").splitlines()
        if len(urls) != 1 or urls[0] not in PUBLISH_ORIGIN_URLS:
            raise PublishError(f"The origin remote must point to {UPDATE_REPOSITORY_URL}.")
    git(root, "var", "GIT_AUTHOR_IDENT")
    git(root, "var", "GIT_COMMITTER_IDENT")


def publication_files(root, manifest_path):
    """Capture and verify the exact bytes that will go into the commit."""
    manifest_data = manifest_path.read_bytes()
    manifest = SetupWizard._validate_manifest(json.loads(manifest_data))
    files = {UPDATE_MANIFEST_PATH: (manifest_data, "100644")}
    for entry in manifest["scripts"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise PublishError(f"Cannot publish a script outside the checkout: {entry['path']}")
        data = path.read_bytes()
        if update_script_sha256(data, entry["preserve"]) != entry["sha256"]:
            raise PublishError(
                f"{entry['path']} changed after preparation. Run the command again before publishing."
            )
        files[entry["path"]] = (data, "100755" if entry["executable"] else "100644")
    return files


def publish_update(root, manifest_path, message):
    """Append one verified commit to remote main without merging the local branch."""
    files = publication_files(root, manifest_path)
    print(f"\nReading the latest {UPDATE_REPOSITORY_URL} ({UPDATE_BRANCH})...", flush=True)
    git(root, "fetch", "--no-tags", "origin", f"refs/heads/{UPDATE_BRANCH}")
    base = git(root, "rev-parse", "--verify", "FETCH_HEAD^{commit}").decode().strip()

    with tempfile.TemporaryDirectory(prefix="xcop_publish_") as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        git(root, "read-tree", base, env=env)
        for relative_path, (data, mode) in files.items():
            blob = git(root, "hash-object", "-w", "--stdin", data=data).decode().strip()
            git(root, "update-index", "--add", "--cacheinfo", f"{mode},{blob},{relative_path}", env=env)
        tree = git(root, "write-tree", env=env).decode().strip()

    if tree == git(root, "rev-parse", f"{base}^{{tree}}").decode().strip():
        print("GitHub already contains these scripts and matching checksums. No upload needed.")
        return base

    changed_paths = git(
        root, "diff-tree", "--no-commit-id", "--name-only", "--no-renames", "-r", "-z", base, tree,
    ).decode("utf-8").rstrip("\0").split("\0")
    commit = git(root, "commit-tree", tree, "-p", base, "-m", message).decode().strip()
    # Retain the commit for inspection/recovery even if authentication or push fails.
    git(root, "update-ref", "refs/xcop-publish/latest", commit)
    print(f"Created commit {commit[:12]} with {len(changed_paths)} changed publication files:", flush=True)
    for relative_path in changed_paths:
        print(f"  {relative_path}", flush=True)
    print("Uploading to GitHub. Complete Git's sign-in if prompted...", flush=True)
    try:
        git(root, "push", "origin", f"{commit}:refs/heads/{UPDATE_BRANCH}")
    except PublishError as exc:
        raise PublishError(
            f"{exc}\nPublication did not complete. Your prepared files and commit {commit} are saved. "
            "If GitHub changed during publishing, run the same command again. "
            "For a sign-in or permission error, sign in to GitHub with Git and retry."
        ) from exc
    print(f"Published successfully: {UPDATE_REPOSITORY_URL}/commit/{commit}")
    print("On the cluster, click Check Again in the script updater.")
    return commit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish", action="store_true",
        help="prepare, commit only changed managed scripts and the manifest, and upload to GitHub",
    )
    parser.add_argument(
        "--message", default="Publish xcop scripts and matching update manifest",
        help="Git commit message used with --publish",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        if args.publish:
            check_publish_repository(root)
        manifest_path = generate_update_manifest(root, clean=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("Removed managed shebangs and cleared xcop's five software selections.")
        print("Scripts use LF line endings. Update-manifest SHA-256 values:")
        for entry in manifest["scripts"]:
            print(f"{entry['sha256']}  {entry['path']}")
        print(f"\nGenerated: {manifest_path}")
        if args.publish:
            publish_update(root, manifest_path, args.message)
        else:
            print("To commit and upload these scripts together, run: python prepare_update.py --publish")
    except (OSError, ValueError, UpdateError, PublishError) as exc:
        print(f"Could not {'publish' if args.publish else 'prepare'} update: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
