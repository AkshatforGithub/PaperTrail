"""Regression tests for the jinja2 security bump.

The repository requires jinja2>=3.1.5 to close GHSA-gmj6-6f8f-6699,
GHSA-q2x7-8rv6-6q7h, PYSEC-2026-1472 and PYSEC-2026-1475. These tests pin
that requirement (so an old, vulnerable install fails) and guard the rendering
path the API uses -- FastAPI's Jinja2Templates over the shipped index.html.

The full app (papertrail.api.main) is deliberately not imported here: it mounts
a static directory that does not exist in a fresh checkout, so importing it
needs extra setup that is unrelated to the jinja2 bump.
"""

from pathlib import Path

import jinja2
from fastapi.templating import Jinja2Templates
from packaging.requirements import Requirement
from packaging.version import Version
from starlette.requests import Request

# Fixed per the advisories above.
_MIN_FIXED_VERSION = "3.1.5"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_TEMPLATE_DIR = _REPO_ROOT / "src" / "papertrail" / "templates"


def _declared_dependencies() -> list[str]:
    """Names+specifiers in the [project].dependencies array."""
    text = _PYPROJECT.read_text()
    declared: list[str] = []
    in_deps = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "dependencies = [":
            in_deps = True
            continue
        if in_deps:
            if stripped == "]":
                break
            if stripped.startswith('"') and stripped.endswith('",'):
                declared.append(stripped[1:-2])
    return declared


def test_jinja2_at_or_above_fixed_version():
    assert Version(jinja2.__version__) >= Version(_MIN_FIXED_VERSION)


def test_declared_jinja2_minimum_is_the_fixed_version():
    reqs = [Requirement(d) for d in _declared_dependencies()]
    matches = [r for r in reqs if r.name.lower() == "jinja2"]
    assert len(matches) == 1, "jinja2 must be declared in pyproject.toml"
    assert matches[0].specifier.contains(_MIN_FIXED_VERSION)


def _index_request() -> Request:
    """A minimal GET / scope, enough for Jinja2Templates.TemplateResponse."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "state": {},
    }
    return Request(scope)


def test_index_renders_via_jinja2_templates():
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    response = templates.TemplateResponse(_index_request(), "index.html")
    body = response.body.decode()
    assert "PaperTrail" in body
    assert '<form id="search-form"' in body
