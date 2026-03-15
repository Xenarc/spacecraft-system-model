"""
watcher.py

Passive agent. Runs alongside the monitor. Inverts control — the system
decides when to involve the LLM, not the user.

Pipeline:

  Postgres (metrics log)
      ↓
  Watcher — polls on interval, runs fast detection rules
      ↓  (only on anomaly)
  LLM — reasons over structured context, produces narrative
      ↓
  OpenWebUI channel API — posts message, appears in operator's feed unprompted

The LLM never sees routine telemetry. It only fires when the fast layer
says something is worth narrating. Detection is cheap and deterministic.
Reasoning is where the LLM earns its place.

Dependencies:
  pip install asyncpg httpx anthropic
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

import anthropic
import asyncpg
import httpx


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class WatcherConfig:
    # Postgres
    pg_dsn: str = "postgresql://user:password@localhost/spacecraft"

    # Anthropic
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

    # OpenWebUI
    openwebui_base_url: str = "http://localhost:3000"
    openwebui_api_key: str = ""         # Settings → Account → API Keys
    openwebui_channel_id: str = ""      # the channel to post alerts into

    # Timing
    poll_interval_s: float = 10.0       # how often to check metrics
    history_window_minutes: int = 15    # context window sent to LLM

    # Spacecraft to watch
    spacecraft: str = "DemoSat-1"


# =============================================================================
# FAST DETECTION LAYER
# =============================================================================
#
# Deterministic rules over the metric snapshot.
# Returns a list of triggered anomalies — each is a dict with name + context.
# The LLM is only called when this list is non-empty.
#
# These are intentionally simple thresholds. Replace with trend detection,
# rate-of-change checks, or statistical bounds as needed.

@dataclass
class Anomaly:
    rule: str
    severity: str           # "warning" | "critical"
    value: float
    threshold: float
    message: str


def detect_anomalies(metrics: dict[str, float]) -> list[Anomaly]:
    anomalies = []

    def check(rule, value, threshold, direction, severity, message):
        triggered = value < threshold if direction == "below" else value > threshold
        if triggered:
            anomalies.append(Anomaly(rule, severity, value, threshold, message))

    check(
        "mission_viable", metrics.get("mission_viable", 1.0),
        threshold=0.5, direction="below", severity="critical",
        message="Mission viability has dropped.",
    )
    check(
        "command_rejection_streak", metrics.get("command_rejection_streak", 0.0),
        threshold=5.0, direction="above", severity="critical",
        message="Sustained command rejections from a subsystem.",
    )
    check(
        "command_acceptance_rate", metrics.get("command_acceptance_rate", 1.0),
        threshold=0.7, direction="below", severity="warning",
        message="Command acceptance rate degraded.",
    )
    check(
        "subsystem_availability_mean", metrics.get("subsystem_availability_mean", 1.0),
        threshold=0.6, direction="below", severity="warning",
        message="Mean subsystem availability below threshold.",
    )
    check(
        "critical_subsystem_down", metrics.get("critical_subsystem_down", 0.0),
        threshold=0.5, direction="above", severity="critical",
        message="A mission-critical subsystem is down.",
    )
    check(
        "propellant_remaining_pct", metrics.get("propellant_remaining_pct", 100.0),
        threshold=20.0, direction="below", severity="warning",
        message="Propellant below 20%.",
    )
    check(
        "power_margin_w", metrics.get("power_margin_w", 999.0),
        threshold=0.0, direction="below", severity="critical",
        message="Power margin negative — load exceeds generation.",
    )

    return anomalies


# =============================================================================
# POSTGRES QUERIES
# =============================================================================

async def fetch_current_metrics(
    conn: asyncpg.Connection,
    spacecraft: str,
) -> dict[str, float]:
    rows = await conn.fetch(
        """
        SELECT name, value FROM spacecraft_metrics_current
        WHERE spacecraft = $1
        """,
        spacecraft,
    )
    return {row["name"]: row["value"] for row in rows}


async def fetch_metric_history(
    conn: asyncpg.Connection,
    spacecraft: str,
    window_minutes: int,
) -> dict[str, list[dict]]:
    """
    Returns recent history per metric name:
      { "delta_v_m_per_s": [{"time": ..., "value": ...}, ...], ... }
    """
    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    rows = await conn.fetch(
        """
        SELECT name, value, time
        FROM spacecraft_metrics_log
        WHERE spacecraft = $1 AND time >= $2
        ORDER BY name, time
        """,
        spacecraft, since,
    )
    history: dict[str, list] = {}
    for row in rows:
        history.setdefault(row["name"], []).append({
            "time": row["time"].isoformat(),
            "value": row["value"],
        })
    return history


async def write_alert(
    conn: asyncpg.Connection,
    spacecraft: str,
    severity: str,
    anomalies: list[Anomaly],
    narrative: str,
) -> int:
    """Write alert to Postgres. Returns the alert id."""
    row = await conn.fetchrow(
        """
        INSERT INTO spacecraft_alerts
            (time, spacecraft, severity, anomalies, narrative, acknowledged)
        VALUES ($1, $2, $3, $4, $5, false)
        RETURNING id
        """,
        datetime.now(timezone.utc),
        spacecraft,
        severity,
        json.dumps([
            {"rule": a.rule, "value": a.value, "threshold": a.threshold, "message": a.message}
            for a in anomalies
        ]),
        narrative,
    )
    return row["id"]


async def ensure_alerts_schema(conn: asyncpg.Connection):
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS spacecraft_alerts (
            id           BIGSERIAL PRIMARY KEY,
            time         TIMESTAMPTZ NOT NULL,
            spacecraft   TEXT        NOT NULL,
            severity     TEXT        NOT NULL,
            anomalies    JSONB       NOT NULL,
            narrative    TEXT        NOT NULL,
            acknowledged BOOLEAN     NOT NULL DEFAULT false
        );
    """)


# =============================================================================
# LLM REASONING LAYER
# =============================================================================
#
# The LLM receives a structured context snapshot and produces a narrative.
# It does not do detection — anomalies are pre-identified by the fast layer.
# Its job is to reason over them: what is likely happening, what matters most,
# what should the operator consider.

def _build_prompt(
    spacecraft: str,
    current: dict[str, float],
    history: dict[str, list[dict]],
    anomalies: list[Anomaly],
) -> str:

    anomaly_lines = "\n".join(
        f"  [{a.severity.upper()}] {a.rule} = {a.value:.2f} (threshold: {a.threshold}) — {a.message}"
        for a in anomalies
    )

    # Summarise history as start/end values for brevity
    history_lines = []
    for name, points in history.items():
        if len(points) >= 2:
            start = points[0]["value"]
            end = points[-1]["value"]
            delta = end - start
            sign = "+" if delta >= 0 else ""
            history_lines.append(f"  {name:<40} {start:.2f} → {end:.2f}  ({sign}{delta:.2f})")
    history_summary = "\n".join(history_lines) if history_lines else "  (no history)"

    current_lines = "\n".join(
        f"  {k:<40} {v:.3f}" for k, v in sorted(current.items())
    )

    return f"""You are a spacecraft operations assistant monitoring {spacecraft}.

TRIGGERED ANOMALIES:
{anomaly_lines}

CURRENT METRICS:
{current_lines}

METRIC TRENDS (last {len(history)} minutes, start → end):
{history_summary}

Describe concisely what is happening, which anomalies are most concerning,
and what the operator should consider. Be direct. Flag if mission recovery
looks possible or if the situation is deteriorating. 2-4 sentences."""


async def call_llm(
    prompt: str,
    config: WatcherConfig,
) -> str:
    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
    response = await client.messages.create(
        model=config.llm_model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# =============================================================================
# OPENWEBUI CHANNEL POSTER
# =============================================================================
#
# OpenWebUI exposes a REST API for posting messages into channels.
# A channel is a persistent chat room — messages appear in the operator's
# feed in real time, unprompted, like a Slack notification.
#
# API endpoint: POST /api/v1/channels/{channel_id}/messages
# Docs: http://localhost:3000/docs (when OpenWebUI is running)

async def post_to_openwebui_channel(
    narrative: str,
    anomalies: list[Anomaly],
    alert_id: int,
    config: WatcherConfig,
):
    severity = max((a.severity for a in anomalies), key=lambda s: s == "critical")
    icon = "🔴" if severity == "critical" else "🟡"

    anomaly_summary = "\n".join(
        f"• **{a.rule}** = {a.value:.2f} (threshold {a.threshold})"
        for a in anomalies
    )

    content = (
        f"{icon} **{config.spacecraft} — {severity.upper()} ALERT** (id: {alert_id})\n\n"
        f"{narrative}\n\n"
        f"**Triggered rules:**\n{anomaly_summary}"
    )

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{config.openwebui_base_url}/api/v1/channels/{config.openwebui_channel_id}/messages",
            headers={
                "Authorization": f"Bearer {config.openwebui_api_key}",
                "Content-Type": "application/json",
            },
            json={"content": content},
            timeout=10.0,
        )
        resp.raise_for_status()


# =============================================================================
# ALERT DEDUPLICATION
# =============================================================================
#
# Avoid spamming the channel with the same anomaly on every poll cycle.
# Simple approach: track which rule names are currently active.
# An alert fires when a new rule triggers that wasn't active before,
# or when a critical rule re-triggers after a quiet period.

class AlertState:
    def __init__(self, quiet_period_s: float = 120.0):
        self._active: dict[str, datetime] = {}   # rule -> first_seen
        self._quiet_period = timedelta(seconds=quiet_period_s)

    def filter_new(self, anomalies: list[Anomaly]) -> list[Anomaly]:
        """Return only anomalies that are new or have re-triggered after quiet period."""
        now = datetime.now(timezone.utc)
        new = []
        for a in anomalies:
            last = self._active.get(a.rule)
            if last is None or (now - last) > self._quiet_period:
                new.append(a)
                self._active[a.rule] = now
        # Clear rules that are no longer triggering
        current_rules = {a.rule for a in anomalies}
        for rule in list(self._active):
            if rule not in current_rules:
                del self._active[rule]
        return new

    @property
    def active_rules(self) -> set[str]:
        return set(self._active.keys())


# =============================================================================
# MAIN WATCHER LOOP
# =============================================================================

async def watcher(config: WatcherConfig):
    """
    Main watcher loop. Runs until cancelled.

    On each cycle:
      1. Fetch current metrics from Postgres
      2. Run fast anomaly detection
      3. Filter to new/re-triggered anomalies (deduplication)
      4. If any: call LLM for narrative, write alert to DB, post to OpenWebUI
    """
    conn = await asyncpg.connect(config.pg_dsn)
    await ensure_alerts_schema(conn)

    alert_state = AlertState(quiet_period_s=120.0)
    print(f"Watcher running for '{config.spacecraft}' — polling every {config.poll_interval_s}s")

    try:
        while True:
            # 1. Fetch current state
            current = await fetch_current_metrics(conn, config.spacecraft)
            if not current:
                await asyncio.sleep(config.poll_interval_s)
                continue

            # 2. Fast detection
            all_anomalies = detect_anomalies(current)

            # 3. Deduplicate
            new_anomalies = alert_state.filter_new(all_anomalies)

            # 4. Only invoke LLM if something new is worth narrating
            if new_anomalies:
                history = await fetch_metric_history(
                    conn, config.spacecraft, config.history_window_minutes
                )
                prompt = _build_prompt(config.spacecraft, current, history, new_anomalies)

                try:
                    narrative = await call_llm(prompt, config)
                except Exception as e:
                    narrative = f"[LLM unavailable: {e}] Anomalies: " + "; ".join(
                        a.message for a in new_anomalies
                    )

                severity = "critical" if any(a.severity == "critical" for a in new_anomalies) else "warning"
                alert_id = await write_alert(conn, config.spacecraft, severity, new_anomalies, narrative)

                try:
                    await post_to_openwebui_channel(narrative, new_anomalies, alert_id, config)
                    print(f"  → alert {alert_id} posted to OpenWebUI channel")
                except Exception as e:
                    print(f"  → alert {alert_id} written to DB but OpenWebUI post failed: {e}")

            await asyncio.sleep(config.poll_interval_s)

    except asyncio.CancelledError:
        pass
    finally:
        await conn.close()


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    config = WatcherConfig(
        pg_dsn="postgresql://user:password@localhost/spacecraft",
        anthropic_api_key="sk-ant-...",
        openwebui_base_url="http://localhost:3000",
        openwebui_api_key="your-openwebui-api-key",
        openwebui_channel_id="your-channel-id",
        spacecraft="DemoSat-1",
        poll_interval_s=10.0,
    )
    asyncio.run(watcher(config))
