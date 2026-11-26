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
    # CI may launch Windows PowerShell with module paths inherited from pwsh.
    # Read both DACLs through .NET so the check also works without Get-Acl.
    script = """
        $ErrorActionPreference = 'Stop'
        $PSModuleAutoLoadingPreference = 'None'
        $access = [System.Security.AccessControl.AccessControlSections]::Access
        function Read-Dacl([string]$path) {
            return [IO.Directory]::GetAccessControl($path, $access).GetSecurityDescriptorSddlForm($access)
        }
        if ((Read-Dacl $env:ACL_REFERENCE) -ne (Read-Dacl $env:ACL_RUN)) { exit 1 }
    """
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        env={**os.environ, "ACL_REFERENCE": str(sibling), "ACL_RUN": str(final)},
        check=True,
    )
