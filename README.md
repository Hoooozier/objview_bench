# ObjView Codebase Overview

`objview_code` is organized into three modules:

- `algorithm/`: C++/CUDA planners, workers, and offline tooling.
- `benchmark/`: Python benchmark runtime, APIs, launcher scripts, and baseline methods.
- `data/`: dataset construction, annotation, filtering, and analysis utilities.

## Environment Setup

Each module has its own README with setup details:

- `algorithm/README.md` (CMake/C++/CUDA/Gurobi stack)
- `benchmark/README.md` (Python + PyTorch + PyTorch3D stack)
- `data/README.md` (Python data-processing stack)

Minimal demo: refer to `benchmark/README.md`.

## Unified Usage Rule

Use each module's README as the source of truth and run script-level help first:

```bash
python <script>.py --help
```

or for C++ binaries:

```bash
./<binary> --help
```

## Sanitization Notes

This code folder is sanitized for release:

- No user-specific absolute paths in maintained source files.
- No Chinese comments/notes in maintained source files.
- Inline long command examples are removed from source code and centralized in README files.

