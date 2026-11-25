import os
import subprocess
from pathlib import Path

import pytest

from elliptic_fraud.runs import new_run


@pytest.mark.skipif(os.name != "nt", reason="Windows directory ACL inheritance")
def test_published_report_inherits_the_same_acl_as_a_normal_sibling(
    tmp_path: Path,
) -> None:
    sibling = tmp_path / "ordinary"
    sibling.mkdir()
    with new_run(tmp_path, "eda") as (staging, final):
        (staging / "result").write_text("complete")
    assert (final / "result").read_text() == "complete"
    # A moved TemporaryDirectory can be readable by the test runner yet
    # inaccessible to a different user who normally has access to the parent.
    script = (
        "$normal = (Get-Acl -LiteralPath $env:ACL_REFERENCE).GetSecurityDescriptorSddlForm('Access'); "
        "$report = (Get-Acl -LiteralPath $env:ACL_RUN).GetSecurityDescriptorSddlForm('Access'); "
        "if ($normal -ne $report) { exit 1 }"
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        env={**os.environ, "ACL_REFERENCE": str(sibling), "ACL_RUN": str(final)},
        check=True,
    )
