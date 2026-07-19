from scripts.train_fourier_dual_net import format_decomp_mask


def test_conditioned_decomposition_summary_accepts_a_textual_mask():
    assert format_decomp_mask("conditioned (per-sample)") == "conditioned (per-sample)"
