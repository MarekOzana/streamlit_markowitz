"""Nox configuration file for managing test sessions and code quality checks."""

import nox


@nox.session(python="3.12")
def tests(session):
    """Run the tests with pytest.

    This session performs the following steps:
    1. Installs dependencies from requirements.txt.
    2. Installs pytest and pytest-cov for running tests and generating code coverage.
    3. Installs the package itself (if needed).
    4. Runs pytest on the 'tests' directory.

    Usage:
    - To run this session, use: `nox -s tests`
    """
    # Install dependencies from requirements.txt
    session.install("-r", "requirements.txt")
    session.install("pytest", "pytest-cov")

    # Run pytest on the 'tests' directory
    session.run("pytest", "tests")

@nox.session(python="3.12")
def ruff(session):
    """Run ruff & black code formatter."""
    # Install black, ruff
    session.install("black", "ruff")

    # Run ruff linter
    session.run("ruff", "check", "src", "scripts", "app.py")
    # Run black linter
    session.run("black", "--check", "src", "scripts", "app.py")

