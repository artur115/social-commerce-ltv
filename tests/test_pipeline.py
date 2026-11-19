"""End to end on a small generated dataset: SQL builds and the data checks hold."""
import duckdb
import pytest

from ltv.pipeline import run_sql
from scripts.generate_data import generate


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw") / "events.parquet"
    generate(n_users=1500, seed=7).to_parquet(raw, index=False)
    c = duckdb.connect()
    run_sql(c, raw)
    return c


def q(con, sql):
    return con.sql(sql).fetchall()[0][0]


def test_dedup_removes_duplicates(con):
    assert q(con, "select count(*) from raw_events") > q(con, "select count(*) from stg_events")
    assert q(con, """select count(*) from (select user_id, session_key, event_name, event_at
                     from stg_events group by all having count(*) > 1)""") == 0


def test_sources_are_lowercase(con):
    assert q(con, "select count(*) from stg_events where source <> lower(source)") == 0


def test_every_order_has_a_touchpath(con):
    assert q(con, "select count(*) from orders where transaction_id not in (select transaction_id from touchpaths)") == 0


def test_ltv_windows_are_monotonic(con):
    assert q(con, "select count(*) from user_ltv where not (ltv_30d <= ltv_90d and ltv_90d <= ltv_180d)") == 0


def test_generator_is_deterministic():
    a, b = generate(200, seed=3), generate(200, seed=3)
    assert a.equals(b)
