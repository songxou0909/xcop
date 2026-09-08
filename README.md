# xcop

`xcop` is personal research software for cryo-EM and structural-biology workflows on a Linux cluster. It is authored and maintained by Songxian Ou while working in Ruben's lab.

## Clone and set up

After this repository is published, clone it and enter the inner `xcop` folder:

```bash
git clone https://github.com/songxou0909/xcop.git
cd xcop
python3 xcop_setup.py
```

The `git clone` command creates one folder named `xcop`; all published project files are inside that folder. The setup wizard requires Conda and installs the Python environment used by `xcop`. Some functions also depend on software modules supplied by the user's cluster. Review the included [quick-reference PDF](README.pdf) before running jobs. The detailed DOCX manual is internal lab material and is intentionally not published.

## Latest snapshot only

This repository intentionally presents only the most up-to-date `xcop` snapshot on its `main` branch. It is not maintained as a public version archive. The branch history may be replaced when the software is updated, so users should make a fresh clone to obtain a later snapshot instead of relying on `git pull`.

## Important: FoldX is not included

FoldX is proprietary, license-restricted third-party software. Its academic license does not permit redistribution or sublicensing, so the FoldX executable and associated distribution files are intentionally excluded from this repository. A disclaimer does not create permission to redistribute them.

Each user who wants the FoldX feature must:

1. Obtain the appropriate FoldX license directly from the [official FoldX licensing page](https://foldxsuite.crg.eu/node/737) and accept its terms.
2. Download FoldX from the official source.
3. Put the downloaded FoldX distribution files directly inside:

   ```text
   xcop/xcop_savings/FoldX/
   ```

   The executable must be directly in that folder. Keep any accompanying resource directories, such as `molecules/`, under the same folder.
4. On Linux, make the FoldX executable runnable if needed:

   ```bash
   chmod u+x xcop/xcop_savings/FoldX/<foldx-executable>
   ```

The repository's research-use statement does not grant a FoldX license. Every user remains responsible for complying with the [FoldX academic license](https://foldxsuite.crg.eu/foldx4-academic-licence) or another applicable FoldX license.

## Other third-party work

This project integrates with and depends on work created by other people and organizations. In particular, the feature labelled `AutoIllustrate` in the interface uses the **Amyloid Illustrator** software by Michael R. Sawaya and David Eisenberg. Its files are not claimed as Songxian Ou's work. See [Third-Party Notices](THIRD_PARTY_NOTICES.md) and the installation note in [`xcop_savings/AutoIllustrate/`](xcop_savings/AutoIllustrate/README.md).

All third-party names, trademarks, software, models, publications, and other materials remain the property of their respective owners and are governed by their own terms. Their mention here is descriptive and does not imply affiliation, sponsorship, endorsement, or transfer of rights.

## Research-use and responsibility statement

This repository is made available for cloning and is intended only for private/internal, non-commercial, non-profit academic research. It is not presented as an official product or release of Ruben's lab or any affiliated institution. See the full [Research Use and Responsibility Notice](RESEARCH_USE_NOTICE.md).

This repository is the work of **Songxian Ou in Ruben's lab**. Any conflict of interest arising specifically from the distribution, maintenance, or use of this repository is attributable solely to **Songxian Ou** and must not be attributed to Ruben, Ruben's lab, other lab members, collaborators, or the affiliated institution. This statement does not replace any disclosure required by an institution, employer, funder, journal, conference, or law, and it does not alter any independently documented relationship.

## No warranty

The materials are provided as-is, without warranties or guarantees. Users are responsible for validating commands, results, data handling, cluster policies, software licenses, and fitness for their intended research.
