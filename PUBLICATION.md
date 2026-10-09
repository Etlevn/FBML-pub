# Public release preparation

The current public source snapshot contains Python scripts and documentation only.
All legacy notebooks and the `TR/` directory have been removed from this release.
Personal filesystem paths and tracked IDE configuration are excluded. Pipelines use
`FBML_DATA_ROOT` for private input data; the loader logs dimensions instead of records.

Ignore rules exclude private datasets, local credentials, model artifacts, and
generated results. Local data and results are preserved on the maintainer's machine
and are excluded from the public source archive.

## Verification

```bash
python scripts/check_publication.py
python scripts/check_publication.py --export
```

The checker scans tracked and new non-ignored files, rejects private artifacts and
all notebook files, and detects common credentials, emails, and personal paths.
It prints finding categories without revealing matched values. It does not detect
every possible secret or determine dataset redistribution rights.

Python sources were parsed successfully. The three configuration
modules were checked with both default and overridden data locations. The checker
was verified against synthetic paths, credentials, datasets, and notebook exclusion.
Model training was not run because the private datasets are supplied separately.

## Git history

The original repository must remain private. Its local branches and remote-tracking
branches contain 35 reachable commits; 102 historical file versions contain absolute
filesystem paths. Old notebooks include saved outputs, and commit metadata contains
a personal author email. These contents remain recoverable in the original history.
No common credential token or private-key pattern was found in the historical blob
scan; that result is not a guarantee that all historical content is publishable.

The chosen release method is a new repository built from the sanitized source. Do
not copy the original `.git` directory, push its branches/tags, or change the original
repository to public. `out/FBML-public-source.zip` contains only checked source files
and no Git history. A separate local repository can be initialized from this archive.
Before publishing, use an appropriate public author identity (for example a GitHub
no-reply email), and choose a license if open-source reuse is intended.

The public repository's earlier commits contain sanitized legacy notebooks with
outputs cleared. Removing notebooks from the current source does not erase those
commits. The current branch and newly generated source archives exclude notebooks;
removing every historical notebook would require a separate history rewrite.
