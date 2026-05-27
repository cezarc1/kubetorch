from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .constants import DATASET_ID, HF_CACHE_KEY, run_scoped_key
from .datasets_io import (
    choose_split,
    disable_video_decoding,
    load_dataset_kwargs,
    manifest_dataset_kwargs,
    summarize_dataset,
)
from .reporting import base_manifest, IssueLedger, publish_file, safe_note, write_json


def _first_row(dataset: Any) -> dict[str, Any]:
    split = choose_split(dataset)
    try:
        row = split[0]
        if isinstance(row, dict):
            return row
    except Exception:
        pass
    return next(iter(split))


def _crop_from_video(
    row: dict[str, Any],
    ledger: IssueLedger,
    *,
    cache_dir: Path | None = None,
) -> Image.Image | None:
    video = row.get("video")
    path = video.get("path") if isinstance(video, dict) else video
    if not path and row.get("file_name"):
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=f"videos/{row['file_name']}",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            ledger.exception(
                category="video",
                summary="failed to download the dataset video referenced by file_name",
                exc=exc,
            )
            return None

    if not path:
        ledger.add(
            category="video",
            severity="medium",
            summary="dataset row did not expose a local video path",
            evidence=f"video field type={type(video).__name__}",
            workaround="use torchcodec decoding or materialize video files explicitly",
        )
        return None

    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        frame_index = int(row.get("frame", 0) or 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"cv2 could not read frame {frame_index} from {path}")

        x_a = int(float(row.get("x_min", 0)))
        y_a = int(float(row.get("y_min", 0)))
        x_b = int(float(row.get("x_max", frame.shape[1])))
        y_b = int(float(row.get("y_max", frame.shape[0])))
        x_min, x_max = sorted((x_a, x_b))
        y_min, y_max = sorted((y_a, y_b))
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(max(x_max, x_min + 1), frame.shape[1])
        y_max = min(max(y_max, y_min + 1), frame.shape[0])
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            raise RuntimeError("bbox crop was empty")
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop)
    except Exception as exc:
        ledger.exception(
            category="video",
            summary="failed to extract a crop from the first dataset video",
            exc=exc,
        )
        return None


def run_bioclip_smoke(
    *,
    output_dir: Path,
    namespace: str,
    split: str | None,
    restore_cache_from_datastore: bool = False,
    sample_rows: int,
    model_name: str,
    streaming: bool = True,
) -> Path:
    import open_clip
    import torch
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "hf-cache"
    ledger = IssueLedger(output_dir / "issues.md")

    if restore_cache_from_datastore:
        try:
            from kubetorch.data_store import DataStoreClient

            DataStoreClient(namespace=namespace).get(
                key=HF_CACHE_KEY,
                dest=cache_dir,
                contents=True,
            )
            safe_note(
                f"Restored Hugging Face cache from kt://{namespace}/{HF_CACHE_KEY}"
            )
        except Exception as exc:
            ledger.exception(
                category="data-store",
                summary="failed to restore Hugging Face cache from Kubetorch data store",
                exc=exc,
            )
            raise

    kwargs = load_dataset_kwargs(cache_dir=cache_dir, split=split, streaming=streaming)
    dataset = load_dataset(**kwargs)
    dataset = disable_video_decoding(dataset)
    row = _first_row(dataset)
    crop = _crop_from_video(row, ledger, cache_dir=cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    species = str(row.get("species") or "wetland bird")
    prompts = [
        f"a photo of a {species}",
        "a photo of a Mallard",
        "a photo of a wetland bird",
    ]
    with torch.no_grad():
        text = tokenizer(prompts).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_norms = text_features.norm(dim=-1).detach().cpu().tolist()

        image_logits = None
        if crop is not None:
            image = preprocess_val(crop).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_logits = (
                (image_features @ text_features.T).squeeze(0).detach().cpu().tolist()
            )

    manifest = base_manifest("bioclip-smoke") | {
        "dataset_id": DATASET_ID,
        "model_name": model_name,
        "load_dataset_kwargs": manifest_dataset_kwargs(kwargs),
        "restore_cache_from_datastore": restore_cache_from_datastore,
        "streaming": streaming,
        "device": device,
        "torch_cuda_available": torch.cuda.is_available(),
        "row": {
            key: row.get(key)
            for key in (
                "frame",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "behavior",
                "behavior_id",
                "species",
                "species_id",
            )
        },
        "dataset_summary": summarize_dataset(dataset, sample_rows=sample_rows),
        "prompts": prompts,
        "text_feature_norms": text_norms,
        "image_logits": image_logits,
        "image_crop_available": crop is not None,
    }
    target = write_json(output_dir / "bioclip_smoke.json", manifest)
    publish_file(
        namespace=namespace,
        key=run_scoped_key("bioclip_smoke.json"),
        path=target,
        name="bioclip-smoke",
        metadata={"model_name": model_name, "dataset_id": DATASET_ID},
    )
    publish_file(
        namespace=namespace,
        key=run_scoped_key("issues.md"),
        path=ledger.path,
        name="issues",
        metadata={"command": "bioclip-smoke"},
    )
    safe_note(
        "BioCLIP smoke completed: "
        f"device={device}, crop_available={crop is not None}, species={species}"
    )
    return target
