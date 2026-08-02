"""Permanently bind ``mrna_editflow`` to source bytes in one checkout.

The formal MK0 entrypoints may run in environments containing stale editable
installs, cached path-entry finders, or ignored timestamp-based bytecode.  This
module therefore installs a highest-priority meta-path finder whose loader
reads and compiles the selected checkout's ``.py`` bytes directly.  It never
consults ``sys.path_importer_cache``, ``sys.path_hooks``, or ``__pycache__``.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterator


_FINDER_MARKER = "mk0_strict_source_bytes_finder_v1"


def _package_module_names(package_name: str) -> tuple[str, ...]:
    prefix = f"{package_name}."
    return tuple(
        name
        for name in tuple(sys.modules)
        if name == package_name or name.startswith(prefix)
    )


class _SourceBytesLoader(importlib.abc.Loader):
    """Load one module from its selected source path without reading bytecode."""

    def __init__(self, source_path: Path, *, is_package: bool) -> None:
        self._source_path = source_path
        self._is_package = is_package

    def create_module(self, _spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        source_bytes = self._source_path.read_bytes()
        code = compile(
            source_bytes,
            str(self._source_path),
            "exec",
            dont_inherit=True,
            optimize=0,
        )
        module.__file__ = str(self._source_path)
        module.__cached__ = None
        if self._is_package:
            module.__path__ = [str(self._source_path.parent)]
        exec(code, module.__dict__)


class _StrictWorktreeFinder(importlib.abc.MetaPathFinder):
    """Resolve one package family only from a fixed repository tree."""

    _mk0_strict_finder_marker = _FINDER_MARKER

    def __init__(self, repo_root: Path, package_name: str) -> None:
        self.repo_root = repo_root
        self.package_name = package_name
        self._prefix = f"{package_name}."

    def _source_for(self, fullname: str) -> tuple[Path, bool]:
        if fullname == self.package_name:
            relative_parts: tuple[str, ...] = ()
        else:
            relative_parts = tuple(fullname[len(self._prefix) :].split("."))
        base = self.repo_root.joinpath(*relative_parts)
        package_init = base / "__init__.py"
        if package_init.is_file():
            return package_init.resolve(strict=True), True
        module_path = base.with_suffix(".py")
        if module_path.is_file():
            return module_path.resolve(strict=True), False
        raise ModuleNotFoundError(
            f"{fullname} is absent from the strictly bound worktree"
        )

    def find_spec(
        self,
        fullname: str,
        _path: object = None,
        _target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self.package_name and not fullname.startswith(self._prefix):
            return None
        source_path, is_package = self._source_for(fullname)
        loader = _SourceBytesLoader(source_path, is_package=is_package)
        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=str(source_path),
            is_package=is_package,
        )
        if spec is None:
            raise RuntimeError(f"could not build a strict source spec for {fullname}")
        if is_package:
            spec.submodule_search_locations = [str(source_path.parent)]
        spec.cached = None
        return spec


def _is_prior_strict_finder(finder: object, package_name: str) -> bool:
    return (
        getattr(finder, "_mk0_strict_finder_marker", None) == _FINDER_MARKER
        and getattr(finder, "package_name", None) == package_name
    )


def _assert_package_family_is_source_bound(
    repo_root: Path,
    package_name: str,
) -> None:
    for name in _package_module_names(package_name):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        loader = getattr(spec, "loader", None)
        if module_file is None or not isinstance(loader, _SourceBytesLoader):
            raise RuntimeError(f"strict import produced an unbound module: {name}")
        try:
            Path(module_file).resolve(strict=True).relative_to(repo_root)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"strict import escaped the current worktree: {name} -> {module_file}"
            ) from error
        if getattr(module, "__cached__", None) is not None:
            raise RuntimeError(f"strict import accepted cached bytecode: {name}")


@contextmanager
def strict_worktree_package_import(
    repo_root: Path,
    package_name: str = "mrna_editflow",
) -> Iterator[ModuleType]:
    """Yield a source-byte-bound package and keep its finder installed.

    The finder remains first in ``sys.meta_path`` after this context exits, so
    later lazy imports in the package family cannot fall back to an editable
    install, a cached path finder, or bytecode.  Repeated installation replaces
    older strict finders for the same package instead of stacking them.
    """

    root = repo_root.expanduser().resolve(strict=True)
    package_init = root / "__init__.py"
    if not package_init.is_file():
        raise RuntimeError(f"package initializer is missing: {package_init}")

    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not _is_prior_strict_finder(finder, package_name)
    ]
    finder = _StrictWorktreeFinder(root, package_name)
    sys.meta_path.insert(0, finder)
    for name in _package_module_names(package_name):
        del sys.modules[name]
    importlib.invalidate_caches()

    try:
        package = importlib.import_module(package_name)
        yield package
        _assert_package_family_is_source_bound(root, package_name)
    except BaseException:
        for name in _package_module_names(package_name):
            del sys.modules[name]
        raise
    finally:
        importlib.invalidate_caches()
