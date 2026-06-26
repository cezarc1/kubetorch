from qwen3_asr_orin.metrics import normalize_transcript, word_error_rate


def test_normalize_transcript_lowercases_punctuation_and_whitespace():
    assert (
        normalize_transcript("  Hello,   Orin!\nThis is Qwen-3.  ")
        == "hello orin this is qwen 3"
    )


def test_word_error_rate_handles_insertions_deletions_and_substitutions():
    reference = "the quick brown fox"
    hypothesis = "the fast brown fox jumps"

    assert word_error_rate(reference, hypothesis) == 0.5


def test_word_error_rate_is_zero_for_empty_reference_and_empty_hypothesis():
    assert word_error_rate("", "") == 0.0
