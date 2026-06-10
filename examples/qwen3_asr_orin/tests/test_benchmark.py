from qwen3_asr_orin.benchmark import BenchmarkSample, summarize_samples


def test_summarize_samples_reports_latency_rtf_throughput_and_errors():
    samples = [
        BenchmarkSample(
            id="ok-1",
            dataset="librispeech",
            language="en",
            duration_seconds=2.0,
            latency_seconds=0.5,
            transcript="hello orin",
            reference="hello orin",
            error=None,
        ),
        BenchmarkSample(
            id="ok-2",
            dataset="fleurs",
            language="fr",
            duration_seconds=4.0,
            latency_seconds=1.0,
            transcript="bonjour orin",
            reference="bonjour orin",
            error=None,
        ),
        BenchmarkSample(
            id="err-1",
            dataset="fleurs",
            language="de",
            duration_seconds=3.0,
            latency_seconds=2.0,
            transcript="",
            reference="hallo orin",
            error="HTTP 500",
        ),
    ]

    summary = summarize_samples(samples, wall_seconds=3.0, model="qwen3-asr", runtime="sglang", precision="fp16")

    assert summary["model"] == "qwen3-asr"
    assert summary["runtime"] == "sglang"
    assert summary["precision"] == "fp16"
    assert summary["attempted"] == 3
    assert summary["completed"] == 2
    assert summary["errors"]["total"] == 1
    assert summary["audio_seconds_completed"] == 6.0
    assert summary["throughput_audio_seconds_per_second"] == 2.0
    assert summary["completed_requests_per_second"] == 2 / 3
    assert summary["latency_seconds"]["p50"] == 0.75
    assert summary["latency_seconds"]["p95"] == 0.975
    assert summary["rtf"]["mean"] == 0.25
    assert summary["wer"]["mean"] == 0.0
