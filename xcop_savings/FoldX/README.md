# Install your licensed FoldX copy here

FoldX is not distributed with `xcop` because its license restricts redistribution and sublicensing.

Each user must obtain an appropriate license and download FoldX directly from the official provider:

- Licensing information: <https://foldxsuite.crg.eu/node/737>
- Academic license: <https://foldxsuite.crg.eu/foldx4-academic-licence>

After downloading it, put the FoldX distribution files in this directory. The final layout should resemble:

```text
xcop/xcop_savings/FoldX/
├── README.md
├── <foldx-executable>
└── molecules/
    └── ...
```

The FoldX executable must be directly inside this directory because `xcop_savings/foldx.py` searches here for a filename containing `foldx`. On Linux, make it executable if necessary:

```bash
chmod u+x <foldx-executable>
```

Do not commit or redistribute the downloaded FoldX files through this repository. Each user is responsible for complying with the FoldX license that applies to them.
