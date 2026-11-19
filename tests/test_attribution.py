import pandas as pd
import pytest

from ltv import attribution

T = pd.Timestamp


def paths():
    rows = [
        # order A: tiktok -> email -> direct, $100
        ("A", 100.0, T("2025-03-10"), "tiktok", T("2025-03-01"), 1),
        ("A", 100.0, T("2025-03-10"), "email", T("2025-03-05"), 2),
        ("A", 100.0, T("2025-03-10"), "direct", T("2025-03-10"), 3),
        # order B: live_stream only, $50
        ("B", 50.0, T("2025-04-02"), "live_stream", T("2025-04-02"), 1),
    ]
    return pd.DataFrame(rows, columns=["transaction_id", "revenue", "ordered_at", "channel", "session_start_at", "touch_number"])


@pytest.mark.parametrize("m", ["last_click", "first_click", "linear", "time_decay", "position_based"])
def test_credit_sums_to_revenue(m):
    assert attribution.rule_based(paths(), m).sum() == pytest.approx(150.0)


def test_last_and_first_click():
    assert attribution.rule_based(paths(), "last_click")["direct"] == 100
    assert attribution.rule_based(paths(), "first_click")["tiktok"] == 100


def test_position_based_split():
    c = attribution.rule_based(paths(), "position_based")
    assert c["tiktok"] == pytest.approx(40) and c["email"] == pytest.approx(20) and c["direct"] == pytest.approx(40)
    assert c["live_stream"] == pytest.approx(50)


def test_time_decay_favours_recent():
    c = attribution.rule_based(paths(), "time_decay")
    assert c["direct"] > c["email"] > c["tiktok"]


def test_markov_sums_to_revenue():
    assert attribution.markov(paths()).sum() == pytest.approx(150.0)


def test_markov_credits_the_channel_every_path_needs():
    rows = []
    for tid, first in [("A", "tiktok"), ("B", "live_stream"), ("C", "email")]:
        rows.append((tid, 10.0, T("2025-05-02"), first, T("2025-05-01"), 1))
        rows.append((tid, 10.0, T("2025-05-02"), "direct", T("2025-05-02"), 2))
    p = pd.DataFrame(rows, columns=paths().columns)
    credit = attribution.markov(p, non_converting=[["tiktok"]] * 5)
    assert credit.idxmax() == "direct"
