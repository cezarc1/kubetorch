from __future__ import annotations

import sys
import types

import numpy as np

from wetlandbirds_shakedown import bioclip_smoke
from wetlandbirds_shakedown.reporting import IssueLedger


class FakeSplit:
    features = {}

    def __iter__(self):
        return iter(
            [
                {
                    "video": None,
                    "frame": 0,
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 1,
                    "y_max": 1,
                    "behavior": "Preening",
                    "behavior_id": 1,
                    "species": "White Wagtail",
                    "species_id": 3,
                }
            ]
        )


def test_first_row_falls_back_when_getitem_returns_iterable_column():
    class IterableColumn:
        pass

    class FakeIterableSplit:
        def __getitem__(self, _index):
            return IterableColumn()

        def __iter__(self):
            return iter([{"species": "White Wagtail"}])

    assert bioclip_smoke._first_row(FakeIterableSplit()) == {"species": "White Wagtail"}


def install_fake_data_store(monkeypatch, client_cls):
    kubetorch_module = types.ModuleType("kubetorch")
    data_store_module = types.ModuleType("kubetorch.data_store")
    data_store_module.DataStoreClient = client_cls
    kubetorch_module.data_store = data_store_module
    monkeypatch.setitem(sys.modules, "kubetorch", kubetorch_module)
    monkeypatch.setitem(sys.modules, "kubetorch.data_store", data_store_module)


def test_bioclip_smoke_defaults_to_streaming_and_no_cache_restore(
    monkeypatch, tmp_path
):
    calls = []

    def fake_load_dataset(**kwargs):
        calls.append(kwargs)
        return FakeSplit()

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class no_grad:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

    class FakeTextFeatures:
        def norm(self, dim, keepdim=False):
            return self

        def __truediv__(self, _other):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [1.0, 1.0, 1.0]

    class FakeTokens:
        def to(self, _device):
            return self

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def encode_text(self, _text):
            return FakeTextFeatures()

    class FakeOpenClip:
        @staticmethod
        def create_model_and_transforms(_model_name):
            return FakeModel(), None, None

        @staticmethod
        def get_tokenizer(_model_name):
            return lambda _prompts: FakeTokens()

    class FakeDataStoreClient:
        def __init__(self, namespace):
            self.namespace = namespace

        def get(self, **kwargs):
            raise AssertionError(
                f"cache restore should be skipped by default: {kwargs}"
            )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    install_fake_data_store(monkeypatch, FakeDataStoreClient)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "open_clip", FakeOpenClip)
    monkeypatch.setattr(bioclip_smoke, "publish_file", lambda **_kwargs: None)

    bioclip_smoke.run_bioclip_smoke(
        output_dir=tmp_path,
        namespace="kubetorch",
        split="train",
        sample_rows=1,
        model_name="fake-model",
    )

    assert calls[0]["streaming"] is True
    assert calls[0]["split"] == "train"


def test_crop_from_video_downloads_hf_file_name(monkeypatch, tmp_path):
    downloads = []

    def fake_hf_hub_download(*, repo_id, repo_type, filename, cache_dir):
        downloads.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "filename": filename,
                "cache_dir": cache_dir,
            }
        )
        path = tmp_path / "001-white_wagtail.mp4"
        path.write_text("fake video")
        return str(path)

    class FakeCapture:
        def __init__(self, path):
            self.path = path

        def set(self, *_args):
            return None

        def read(self):
            return True, np.ones((4, 4, 3), dtype=np.uint8)

        def release(self):
            return None

    fake_cv2 = types.SimpleNamespace(
        CAP_PROP_POS_FRAMES=1,
        COLOR_BGR2RGB=2,
        VideoCapture=FakeCapture,
        cvtColor=lambda frame, _mode: frame,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    crop = bioclip_smoke._crop_from_video(
        {
            "file_name": "001-white_wagtail.mp4",
            "frame": 0,
            "x_min": 0,
            "y_min": 0,
            "x_max": 2,
            "y_max": 2,
        },
        IssueLedger(tmp_path / "issues.md"),
        cache_dir=tmp_path / "hf-cache",
    )

    assert crop is not None
    assert crop.size == (2, 2)
    assert downloads == [
        {
            "repo_id": "academic-datasets/Visual-WetlandBirds-Dataset",
            "repo_type": "dataset",
            "filename": "videos/001-white_wagtail.mp4",
            "cache_dir": tmp_path / "hf-cache",
        }
    ]
