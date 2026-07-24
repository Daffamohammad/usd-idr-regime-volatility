"""Fallback notebook executor for environments where Jupyter kernels cannot launch.

It executes Python code cells sequentially in the project's active interpreter,
stores stream output and execution counts in-place, and fails on the first cell
error. It is intentionally small: normal users can use nbconvert; this runner
exists for reproducible CI/sandbox validation when local kernel launch is blocked.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import traceback
from pathlib import Path

import nbformat


def execute_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__", "__file__": str(path.resolve())}
    execution_count = 1
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        buffer = io.StringIO()
        cell.outputs = []
        cell.execution_count = execution_count
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(cell.source, f"{path}:cell-{index + 1}", "exec"), namespace)
        except Exception as exc:
            trace = traceback.format_exc()
            cell.outputs = [
                nbformat.v4.new_output(
                    "error",
                    ename=type(exc).__name__,
                    evalue=str(exc),
                    traceback=trace.splitlines(),
                )
            ]
            nbformat.write(notebook, path)
            raise RuntimeError(f"Notebook gagal pada code cell {index + 1}: {exc}") from exc
        output = buffer.getvalue()
        if output:
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=output)]
        execution_count += 1
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    execute_notebook(args.notebook)
    print(f"Executed: {args.notebook}")


if __name__ == "__main__":
    main()
