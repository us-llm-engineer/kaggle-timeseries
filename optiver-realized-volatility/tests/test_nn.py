import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from opt14.nn_models import V3, V4, pct_loss

torch.manual_seed(0)


def test_v3_uses_every_input():
    """Perturbing the features, the sequence, the stock id or the reference changes the output (no input is silently ignored)."""
    m = V3(20).eval()
    x, seq = torch.randn(16, 20), torch.randn(16, 120, 13)
    st, ref = torch.randint(0, 112, (16,)), torch.randn(16)
    base = m(x, seq, st, ref)
    assert (m(x + 1.0, seq, st, ref) - base).abs().max() > 1e-6
    assert (m(x, seq + 1.0, st, ref) - base).abs().max() > 1e-6
    assert (m(x, seq, (st + 1) % 112, ref) - base).abs().max() > 1e-6
    assert torch.allclose(m(x, seq, st, ref + 1.0) - base, torch.ones(16), atol=1e-5)      # exact skip connection from log rv_ref


def test_v4_permutation_equivariant_and_padding_invariant():
    m = V4(20).eval()
    B, S = 3, 10
    x, st, ref = torch.randn(B, S, 20), torch.randint(0, 112, (B, S)), torch.randn(B, S)
    pad = torch.zeros(B, S, dtype=torch.bool)
    out = m(x, st, ref, pad)
    perm = torch.randperm(S)
    assert torch.allclose(m(x[:, perm], st[:, perm], ref[:, perm], pad)[:, :], out[:, perm], atol=1e-4)      # equivariance over stocks
    # padded stocks must not influence the real ones
    x2 = torch.cat([x, torch.randn(B, 4, 20)], 1); st2 = torch.cat([st, torch.zeros(B, 4, dtype=torch.long)], 1)
    ref2 = torch.cat([ref, torch.zeros(B, 4)], 1); pad2 = torch.cat([pad, torch.ones(B, 4, dtype=torch.bool)], 1)
    assert torch.allclose(m(x2, st2, ref2, pad2)[:, :S], out, atol=1e-4)
    # cross-stock attention is live: changing another stock's features changes this stock's output
    x3 = x.clone(); x3[:, 1] += 3.0
    assert (m(x3, st, ref, pad)[:, 0] - out[:, 0]).abs().max() > 1e-6


def test_gradients_reach_every_parameter_group():
    for m, args in ((V3(20), (torch.randn(8, 20), torch.randn(8, 120, 13), torch.randint(0, 112, (8,)), torch.randn(8))),
                    (V4(20), (torch.randn(2, 6, 20), torch.randint(0, 112, (2, 6)), torch.randn(2, 6), torch.zeros(2, 6, dtype=torch.bool)))):
        m.train()
        loss = m(*args).float().exp().mean(); loss.backward()
        dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        assert not dead, dead


def test_pct_loss_matches_definition():
    y = torch.tensor([0.002, 0.004]); lp = torch.log(y * 1.1)
    assert abs(float(pct_loss(lp, y)) - 0.01) < 1e-5            # (0.1)^2


def test_dropout_changes_train_output_but_not_eval():
    m = V3(10, drop=0.5); x, seq, st, ref = torch.randn(32, 10), torch.randn(32, 120, 13), torch.randint(0, 112, (32,)), torch.randn(32)
    m.eval(); a, b = m(x, seq, st, ref), m(x, seq, st, ref)
    assert torch.equal(a, b)
    m.train(); assert not torch.equal(m(x, seq, st, ref), m(x, seq, st, ref))
