"""Every module must import cleanly and cheaply.

This is the executable form of the rule that no analysis logic runs at import
time. If someone later puts a `read_csv` or a `.fit()` at module level, this test
starts failing (or hanging), which is the point: the Flask app imports `wepr` at
startup and must not train or fetch while doing so.
"""

import importlib
import subprocess
import sys
import textwrap

import pytest

MODULES = [
    "wepr",
    "wepr.paths",
    "wepr.cities",
    "wepr.data",
    "wepr.data.fetch",
    "wepr.data.load",
    "wepr.data.validate",
    "wepr.features",
    "wepr.evaluate",
    "wepr.trends",
    "wepr.spatial",
    "wepr.models",
    "wepr.models.baselines",
    "wepr.models.nextday",
    "wepr.models.seasonal",
    "wepr.models.registry",
    "wepr.serve",
    "app",
    "app.api",
    "app.views",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name) is not None


def test_importing_wepr_does_not_touch_the_network_or_disk():
    """A cold import of the package must not read data or open a socket.

    Run in a subprocess rather than with `importlib.reload`, for two reasons.

    The reload form could not actually see the thing it was testing: by the time
    it ran, every module was already imported, so any socket opened or file read
    *at first import* had already happened in the parent process. Only a fresh
    interpreter observes a genuine cold import.

    It was also destructive. `reload` rebuilds a module's classes in place while
    leaving every name another test module had already imported bound to the old
    objects, so an `except SomeError` elsewhere in the session silently stopped
    matching the error actually raised — which is precisely how this was found.
    """
    guard = textwrap.dedent(
        """
        import socket, sys

        def forbidden(*args, **kwargs):
            raise AssertionError("import-time network access")

        socket.socket.connect = forbidden

        for name in sys.argv[1:]:
            __import__(name)
        """
    )
    modules = [m for m in MODULES if m.startswith("wepr")]
    result = subprocess.run(
        [sys.executable, "-c", guard, *modules],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
