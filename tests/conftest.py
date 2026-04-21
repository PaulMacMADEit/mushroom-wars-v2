from __future__ import annotations

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    row = getattr(item, "accuracy_row", None)
    if row is None:
        return

    status = "Pass" if report.passed else "Fail"
    row["status"] = status
    report.accuracy_row = row


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter):
    rows = []
    for outcome in ("passed", "failed", "xfailed"):
        for report in terminalreporter.stats.get(outcome, []):
            row = getattr(report, "accuracy_row", None)
            if row is not None:
                rows.append(row)

    if not rows:
        return

    terminalreporter.write_sep("=", "accuracy fixtures")
    terminalreporter.write_line("| Test | Setup | Expected outcome | Actual outcome | Status |")
    terminalreporter.write_line("|---|---|---|---|---|")
    for row in rows:
        terminalreporter.write_line(
            f"| {row['name']} | {row['setup']} | {row['expected']} | {row['actual']} | {row['status']} |"
        )
