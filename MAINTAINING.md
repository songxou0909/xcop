# Maintaining the single current snapshot

Git requires at least one commit. This repository is therefore maintained as one current commit on `main`, rather than as a sequence of public versions.

## Publish an update

1. Replace or edit the files in the local `xcop` folder.
2. Regenerate `xcop_savings/update_manifest.json` so the updater can verify the current public scripts:

   ```bash
   python3 xcop_setup.py --generate-update-manifest
   ```

3. Confirm that restricted FoldX and Amyloid Illustrator files remain ignored.
4. Amend the existing commit and replace the remote branch:

   ```bash
   git add -A
   git commit --amend --no-edit
   git push --force-with-lease origin main
   ```

This leaves one commit reachable from the public `main` branch. Rewriting history means existing clones will no longer update cleanly; users should make a fresh clone after a snapshot replacement.

## Important limitation

Rewriting a public Git branch is not a guarantee that an older published file has been erased. GitHub caches, forks, downloaded archives, and other people's clones may retain it. Review private, licensed, and sensitive material before every publication.
