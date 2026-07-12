from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALLATION = REPO_ROOT / "python_client/kubetorch/docs/start/installation.md"


def test_installation_documents_cluster_and_nvidia_prerequisites():
    guide = INSTALLATION.read_text()
    normalized = " ".join(guide.split())

    assert "existing Kubernetes or k3s cluster" in guide
    assert "GPU Operator is not included" in guide
    assert "nvidia-device-plugin.enabled=true" in guide
    assert "dcgm-exporter.enabled=false" in guide
    assert "drivers and container runtime" in guide
    assert "Amazon EKS, Azure AKS, Google GKE, and k3s homelabs" in normalized
    assert "```{important}" not in guide
