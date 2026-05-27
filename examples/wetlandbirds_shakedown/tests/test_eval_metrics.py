from wetlandbirds_shakedown.eval_metrics import (
    compute_classification_metrics,
    summarize_performance,
)


def test_compute_classification_metrics_reports_topk_macro_f1_and_recall():
    predictions = [
        {
            "true_species": "Mallard",
            "ranked_species": ["Mallard", "Gadwall", "Coot"],
            "crop_available": True,
        },
        {
            "true_species": "Mallard",
            "ranked_species": ["Gadwall", "Mallard", "Coot"],
            "crop_available": True,
        },
        {
            "true_species": "Gadwall",
            "ranked_species": ["Mallard", "Gadwall", "Coot"],
            "crop_available": True,
        },
        {
            "true_species": "Coot",
            "ranked_species": [],
            "crop_available": False,
            "error": "crop unavailable",
        },
    ]

    metrics = compute_classification_metrics(predictions, samples_requested=4)

    assert metrics["samples_requested"] == 4
    assert metrics["samples_evaluated"] == 3
    assert metrics["crop_success_rate"] == 0.75
    assert metrics["top1_accuracy"] == 1 / 3
    assert metrics["top3_accuracy"] == 1.0
    assert metrics["macro_f1"] == 0.25
    assert metrics["per_class_counts"] == {"Gadwall": 1, "Mallard": 2}
    assert metrics["per_class_recall"] == {"Gadwall": 0.0, "Mallard": 0.5}


def test_summarize_performance_reports_throughput_and_cuda_memory():
    performance = summarize_performance(
        samples_evaluated=5,
        timings={
            "dataset_load_seconds": 1.0,
            "model_load_seconds": 2.0,
            "crop_decode_seconds_total": 0.5,
            "inference_seconds_total": 2.5,
        },
        device_info={
            "device": "cuda",
            "cuda_available": True,
            "cuda_device_name": "NVIDIA GeForce RTX 4090",
            "cuda_peak_memory_mb": 512.25,
        },
    )

    assert performance["samples_per_second"] == 2.0
    assert performance["device"] == "cuda"
    assert performance["cuda_device_name"] == "NVIDIA GeForce RTX 4090"
    assert performance["cuda_peak_memory_mb"] == 512.25
