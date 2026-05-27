from wetlandbirds_shakedown.datasets_io import (
    DATASET_ID,
    disable_video_decoding,
    load_dataset_kwargs,
    summarize_dataset,
)


def test_load_dataset_kwargs_defaults_to_full_dataset(tmp_path):
    kwargs = load_dataset_kwargs(cache_dir=tmp_path / "hf-cache")

    assert kwargs == {
        "path": DATASET_ID,
        "cache_dir": str(tmp_path / "hf-cache"),
    }


def test_load_dataset_kwargs_supports_split_and_streaming(tmp_path):
    kwargs = load_dataset_kwargs(
        cache_dir=tmp_path / "hf-cache",
        split="train",
        streaming=True,
    )

    assert kwargs["path"] == DATASET_ID
    assert kwargs["split"] == "train"
    assert kwargs["streaming"] is True
    assert "features" in kwargs
    assert "file_name" in kwargs["features"]
    assert "video" not in kwargs["features"]


def test_summarize_dataset_handles_dataset_dict_like_objects():
    class FakeSplit:
        column_names = ["video", "frame", "species", "behavior"]

        def __len__(self):
            return 12

        def select(self, _indices):
            return self

        def __iter__(self):
            return iter(
                [
                    {"species": "White Wagtail", "behavior": "Preening"},
                    {"species": "Mallard", "behavior": "Walking"},
                    {"species": "Mallard", "behavior": "Walking"},
                ]
            )

    summary = summarize_dataset({"train": FakeSplit()}, sample_rows=3)

    assert summary["splits"]["train"]["rows"] == 12
    assert summary["splits"]["train"]["columns"] == [
        "video",
        "frame",
        "species",
        "behavior",
    ]
    assert summary["splits"]["train"]["sample_species"] == ["Mallard", "White Wagtail"]
    assert summary["splits"]["train"]["sample_behaviors"] == ["Preening", "Walking"]


def test_disable_video_decoding_casts_video_columns():
    calls = []

    class FakeVideo:
        def __init__(self, decode=True):
            self.decode = decode

    class FakeSplit:
        features = {"video": FakeVideo(), "species": object()}

        def cast_column(self, name, feature):
            calls.append((name, feature.decode))
            return self

    dataset = {"train": FakeSplit()}

    assert disable_video_decoding(dataset, video_cls=FakeVideo) is dataset
    assert calls == [("video", False)]
