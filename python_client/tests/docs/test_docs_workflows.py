from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _workflow(name: str) -> dict:
    return yaml.safe_load((REPO_ROOT / ".github/workflows" / name).read_text())


def test_tutorial_smoke_only_runs_main_in_protected_environment():
    smoke = _workflow("docs_tutorial_smoke.yaml")["jobs"]["smoke"]

    assert smoke["if"] == "github.ref == 'refs/heads/main'"
    assert smoke["runs-on"] == "ubuntu-latest"
    assert smoke["environment"] == "tutorial-validation"
    assert "self-hosted" not in str(smoke)
    configure = next(step for step in smoke["steps"] if step["name"] == "Configure cluster credentials")
    assert configure["env"] == {"KUBECONFIG_B64": "${{ secrets.KUBECONFIG_B64 }}"}


def test_pages_write_permissions_are_limited_to_deploy_job():
    workflow = _workflow("build_docs.yaml")

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["build-docs"]["permissions"] == {
        "contents": "read",
        "pages": "read",
    }
    assert workflow["jobs"]["deploy-docs"]["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
