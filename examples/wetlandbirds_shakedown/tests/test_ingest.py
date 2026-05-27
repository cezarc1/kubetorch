from __future__ import annotations

import sys
import types

from wetlandbirds_shakedown import ingest


class FakeSplit:
    column_names = ["video", "frame", "species", "behavior"]
    features = {}

    def __iter__(self):
        return iter(
            [
                {"species": "White Wagtail", "behavior": "Preening"},
                {"species": "Mallard", "behavior": "Walking"},
            ]
        )


def install_fake_data_store(monkeypatch, client_cls):
    kubetorch_module = types.ModuleType("kubetorch")
    data_store_module = types.ModuleType("kubetorch.data_store")
    data_store_module.DataStoreClient = client_cls
    kubetorch_module.data_store = data_store_module
    monkeypatch.setitem(sys.modules, "kubetorch", kubetorch_module)
    monkeypatch.setitem(sys.modules, "kubetorch.data_store", data_store_module)


def test_ingest_defaults_to_streaming_and_skips_cache_upload(monkeypatch, tmp_path):
    calls = []
    published = []

    def fake_load_dataset(**kwargs):
        calls.append(kwargs)
        return FakeSplit()

    class FakeDataStoreClient:
        def __init__(self, namespace):
            self.namespace = namespace

        def put(self, **kwargs):
            raise AssertionError(
                f"cache upload should be skipped in streaming mode: {kwargs}"
            )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    install_fake_data_store(monkeypatch, FakeDataStoreClient)
    monkeypatch.setattr(
        ingest,
        "publish_file",
        lambda **kwargs: published.append(kwargs),
    )

    target = ingest.run_ingest_hf(
        output_dir=tmp_path,
        namespace="kubetorch",
        split="train",
        sync_cache_to_datastore=True,
        sample_rows=2,
    )

    assert target == tmp_path / "download_manifest.json"
    assert calls[0]["streaming"] is True
    assert calls[0]["split"] == "train"
    assert len(published) == 2


def test_materialized_ingest_can_sync_cache(monkeypatch, tmp_path):
    uploads = []

    def fake_load_dataset(**_kwargs):
        return FakeSplit()

    class FakeDataStoreClient:
        def __init__(self, namespace):
            self.namespace = namespace

        def put(self, **kwargs):
            uploads.append({"namespace": self.namespace, **kwargs})

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    install_fake_data_store(monkeypatch, FakeDataStoreClient)
    monkeypatch.setattr(ingest, "publish_file", lambda **_kwargs: None)

    ingest.run_ingest_hf(
        output_dir=tmp_path,
        namespace="kubetorch",
        split=None,
        sync_cache_to_datastore=True,
        sample_rows=1,
        streaming=False,
    )

    assert uploads == [
        {
            "namespace": "kubetorch",
            "key": "datasets/visual-wetlandbirds-shakedown/hf-cache",
            "src": tmp_path / "hf-cache",
            "contents": True,
            "force": True,
        }
    ]
