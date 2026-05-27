from pathlib import Path


def test_controller_pins_kubernetes_client_below_36():
    requirements = Path(__file__).parents[1] / "requirements.txt"
    lines = requirements.read_text().splitlines()

    assert "kubernetes>=34.1.0,<36.0.0" in lines
