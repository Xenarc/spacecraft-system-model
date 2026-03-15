"""
spacecraft.py

Pipeline:

  async streams → queue → reducer → SpacecraftState
                                          ↓
                              rules(Spacecraft, SpacecraftState)
                                          ↓
                                      metrics → Postgres

Five concerns, cleanly separated:

  1. System model      — physical spacecraft, pure data, frozen, loaded once
  2. Operational state — what the spacecraft is doing right now, mutable via reducer
  3. Events            — typed messages arriving from async streams
  4. Rules             — plain functions over (Spacecraft, SpacecraftState)
  5. Monitor           — async consumer: reduce → evaluate → append to Postgres

The system model never changes during a run.
The operational state changes on every event.
Rules draw from both — physics from the model, current conditions from the state.

Dependencies:
  pip install asyncpg
"""

from __future__ import annotations

import asyncio
import math
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import asyncpg


# =============================================================================
# 1. SYSTEM MODEL — physical spacecraft, frozen, loaded once
# =============================================================================

@dataclass(frozen=True)
class Propellant:
    name: str
    density_kg_per_m3: float

@dataclass(frozen=True)
class Propulsion:
    thrust_n: float
    isp_s: float
    propellant: Propellant
    dry_mass_kg: float
    propellant_mass_kg: float

@dataclass(frozen=True)
class Power:
    generation_w: float
    storage_wh: float
    dry_mass_kg: float

@dataclass(frozen=True)
class Avionics:
    cpu: str
    sensors: tuple[str, ...]
    average_power_w: float
    dry_mass_kg: float

@dataclass(frozen=True)
class ThermalControl:
    method: str
    dry_mass_kg: float

@dataclass(frozen=True)
class Structure:
    material: str
    dry_mass_kg: float

@dataclass(frozen=True)
class Payload:
    name: str
    dry_mass_kg: float
    average_power_w: float = 0.0

@dataclass(frozen=True)
class Spacecraft:
    name: str
    structure: Structure
    propulsion: Propulsion
    power: Power
    avionics: Avionics
    thermal: ThermalControl
    payloads: tuple[Payload, ...] = field(default_factory=tuple)


# =============================================================================
# 2. OPERATIONAL STATE
# =============================================================================
#
# Describes what the spacecraft *is doing*.
# Not frozen — reducer deepcopies at the boundary and mutates freely inside.
#
# NOTE: if state becomes deeply nested or updates become sparse and targeted,
# replace deepcopy with composed lenses:
#   Lens[S, A] = (get: S -> A, set: S -> A -> S)
# Compose lenses for nested access — only the touched path is reallocated,
# not the whole tree.

COMMAND_WINDOW = 50

@dataclass
class LivePropulsion:
    # Mutable shadow of Propulsion holding live measured values.
    # None means no live data yet — use model nominal.
    propellant_mass_kg: float | None = None

@dataclass(frozen=True)
class Position:
    x_m: float
    y_m: float
    z_m: float

@dataclass(frozen=True)
class Task:
    name: str
    required_subsystems: tuple[str, ...]
    index: int

@dataclass(frozen=True)
class Mission:
    name: str
    tasks: tuple[Task, ...]

@dataclass(frozen=True)
class CommandEvent:
    subsystem: str
    command: str
    accepted: bool
    latency_ms: float
    time: datetime

@dataclass
class SpacecraftState:
    position: Position              = field(default_factory=lambda: Position(0.0, 0.0, 0.0))
    position_time: datetime         = field(default_factory=lambda: datetime.now(timezone.utc))
    mission: Mission | None         = None
    current_task: Task | None       = None
    subsystems: dict[str, float]    = field(default_factory=dict)   # name -> availability [0,1]
    commands: list[CommandEvent]    = field(default_factory=list)   # bounded to COMMAND_WINDOW
    propulsion: LivePropulsion       = field(default_factory=LivePropulsion)


# =============================================================================
# 3. EVENTS
# =============================================================================

@dataclass(frozen=True)
class PositionUpdate:
    position: Position
    time: datetime

@dataclass(frozen=True)
class MissionUpdate:
    mission: Mission
    current_task: Task

@dataclass(frozen=True)
class SubsystemUpdate:
    subsystem: str
    availability: float

@dataclass(frozen=True)
class CommandUpdate:
    event: CommandEvent

@dataclass(frozen=True)
class PropellantUpdate:
    # Live propellant mass from flow meter or estimator.
    # Bridge between streams and the frozen system model —
    # rules prefer this over the nominal model value when present.
    propellant_mass_kg: float

Event = PositionUpdate | MissionUpdate | SubsystemUpdate | CommandUpdate | PropellantUpdate


# =============================================================================
# REDUCER
# =============================================================================

def reduce(state: SpacecraftState, event: Event) -> SpacecraftState:
    s = deepcopy(state)

    match event:

        case PositionUpdate(position=p, time=t):
            s.position = p
            s.position_time = t

        case MissionUpdate(mission=m, current_task=task):
            s.mission = m
            s.current_task = task

        case SubsystemUpdate(subsystem=name, availability=a):
            s.subsystems[name] = a

        case CommandUpdate(event=cmd):
            s.commands.append(cmd)
            s.commands = s.commands[-COMMAND_WINDOW:]

        case PropellantUpdate(propellant_mass_kg=mass):
            s.propulsion.propellant_mass_kg = mass

        case _:
            pass

    return s


# =============================================================================
# 4. RULES
# =============================================================================
#
# Each rule takes (Spacecraft, SpacecraftState) and returns a float.
# sc    — fixed physical properties (isp, dry mass, power generation)
# state — live operational values (position, subsystem health, commands)
#
# Rules compose by calling each other directly.
# Add entries to RULES to expose as metrics.

# ---- helpers ----

def _live_propellant_mass(sc: Spacecraft, state: SpacecraftState) -> float:
    """Live propellant mass from state if available, else nominal model value."""
    if state.propulsion.propellant_mass_kg is not None:
        return state.propulsion.propellant_mass_kg
    return sc.propulsion.propellant_mass_kg

# ---- propulsion ----

def wet_mass(sc: Spacecraft, state: SpacecraftState) -> float:
    return (
        sc.structure.dry_mass_kg
        + sc.propulsion.dry_mass_kg
        + sc.power.dry_mass_kg
        + sc.avionics.dry_mass_kg
        + sc.thermal.dry_mass_kg
        + sum(p.dry_mass_kg for p in sc.payloads)
        + _live_propellant_mass(sc, state)
    )

def dry_mass(sc: Spacecraft, state: SpacecraftState) -> float:
    return (
        sc.structure.dry_mass_kg
        + sc.propulsion.dry_mass_kg
        + sc.power.dry_mass_kg
        + sc.avionics.dry_mass_kg
        + sc.thermal.dry_mass_kg
        + sum(p.dry_mass_kg for p in sc.payloads)
    )

def delta_v(sc: Spacecraft, state: SpacecraftState) -> float:
    m0, mf = wet_mass(sc, state), dry_mass(sc, state)
    if mf <= 0 or m0 <= mf:
        return 0.0
    return sc.propulsion.isp_s * 9.80665 * math.log(m0 / mf)

def thrust_to_weight(sc: Spacecraft, state: SpacecraftState) -> float:
    return sc.propulsion.thrust_n / (wet_mass(sc, state) * 9.80665)

def propellant_remaining_pct(sc: Spacecraft, state: SpacecraftState) -> float:
    nominal = sc.propulsion.propellant_mass_kg
    return (_live_propellant_mass(sc, state) / nominal * 100.0) if nominal > 0 else 0.0


# ---- power ----

def _total_load_w(sc: Spacecraft) -> float:
    return sc.avionics.average_power_w + sum(p.average_power_w for p in sc.payloads)

def power_margin(sc: Spacecraft, state: SpacecraftState) -> float:
    return sc.power.generation_w - _total_load_w(sc)

def power_endurance(sc: Spacecraft, state: SpacecraftState) -> float:
    return sc.power.storage_wh / max(_total_load_w(sc), 0.001)


# ---- position ----

def distance_from_origin(sc: Spacecraft, state: SpacecraftState) -> float:
    p = state.position
    return math.sqrt(p.x_m**2 + p.y_m**2 + p.z_m**2)


# ---- mission ----

def mission_progress_pct(sc: Spacecraft, state: SpacecraftState) -> float:
    if not state.mission or not state.current_task:
        return 0.0
    total = len(state.mission.tasks)
    return (state.current_task.index / total * 100.0) if total else 0.0

def task_subsystem_conflict(sc: Spacecraft, state: SpacecraftState) -> float:
    if not state.current_task:
        return 0.0
    return 1.0 if any(
        state.subsystems.get(s, 0.0) < 0.5
        for s in state.current_task.required_subsystems
    ) else 0.0

def subsystems_available_count(sc: Spacecraft, state: SpacecraftState) -> float:
    return float(sum(1 for a in state.subsystems.values() if a >= 0.9))

def subsystem_availability_mean(sc: Spacecraft, state: SpacecraftState) -> float:
    return sum(state.subsystems.values()) / len(state.subsystems) if state.subsystems else 0.0

def critical_subsystem_down(sc: Spacecraft, state: SpacecraftState) -> float:
    if not state.current_task:
        return 0.0
    return 1.0 if any(
        state.subsystems.get(s, 0.0) == 0.0
        for s in state.current_task.required_subsystems
    ) else 0.0


# ---- commands ----

def command_acceptance_rate(sc: Spacecraft, state: SpacecraftState) -> float:
    if not state.commands:
        return 1.0
    return sum(c.accepted for c in state.commands) / len(state.commands)

def command_latency_mean_ms(sc: Spacecraft, state: SpacecraftState) -> float:
    if not state.commands:
        return 0.0
    return sum(c.latency_ms for c in state.commands) / len(state.commands)

def command_rejection_streak(sc: Spacecraft, state: SpacecraftState) -> float:
    streak = 0
    for cmd in reversed(state.commands):
        if not cmd.accepted:
            streak += 1
        else:
            break
    return float(streak)

def operational_tempo(sc: Spacecraft, state: SpacecraftState) -> float:
    if len(state.commands) < 2:
        return 0.0
    span_s = (state.commands[-1].time - state.commands[0].time).total_seconds()
    return len(state.commands) / span_s if span_s > 0 else 0.0

def command_vs_availability_mismatch(sc: Spacecraft, state: SpacecraftState) -> float:
    if not state.commands:
        return 0.0
    recent = state.commands[-10:]
    mismatch = sum(1 for c in recent if state.subsystems.get(c.subsystem, 1.0) < 0.7)
    return float(mismatch) / len(recent)


# ---- cross-cutting ----

def mission_viable(sc: Spacecraft, state: SpacecraftState) -> float:
    """Combines physical and operational conditions."""
    if delta_v(sc, state) < 50.0:           return 0.0
    if power_margin(sc, state) < 0.0:       return 0.0
    if task_subsystem_conflict(sc, state):  return 0.0
    if command_rejection_streak(sc, state) > 5: return 0.0
    if subsystem_availability_mean(sc, state) < 0.5: return 0.0
    return 1.0


Rule = Callable[[Spacecraft, SpacecraftState], float]

RULES: list[tuple[str, Rule]] = [
    ("wet_mass_kg",                      wet_mass),
    ("dry_mass_kg",                      dry_mass),
    ("delta_v_m_per_s",                  delta_v),
    ("thrust_to_weight",                 thrust_to_weight),
    ("propellant_remaining_pct",         propellant_remaining_pct),
    ("power_margin_w",                   power_margin),
    ("power_endurance_h",                power_endurance),
    ("distance_from_origin_m",           distance_from_origin),
    ("mission_progress_pct",             mission_progress_pct),
    ("task_subsystem_conflict",          task_subsystem_conflict),
    ("subsystems_available_count",       subsystems_available_count),
    ("subsystem_availability_mean",      subsystem_availability_mean),
    ("critical_subsystem_down",          critical_subsystem_down),
    ("command_acceptance_rate",          command_acceptance_rate),
    ("command_latency_mean_ms",          command_latency_mean_ms),
    ("command_rejection_streak",         command_rejection_streak),
    ("operational_tempo",                operational_tempo),
    ("command_vs_availability_mismatch", command_vs_availability_mismatch),
    ("mission_viable",                   mission_viable),
]

def evaluate_rules(sc: Spacecraft, state: SpacecraftState) -> dict[str, float]:
    results = {}
    for name, rule in RULES:
        try:
            results[name] = rule(sc, state)
        except Exception as e:
            print(f"  rule error [{name}]: {e}")
    return results


# =============================================================================
# 5. POSTGRES — append-only log + current state view
# =============================================================================

CREATE_LOG = """
CREATE TABLE IF NOT EXISTS spacecraft_metrics_log (
    id          BIGSERIAL        PRIMARY KEY,
    time        TIMESTAMPTZ      NOT NULL,
    spacecraft  TEXT             NOT NULL,
    name        TEXT             NOT NULL,
    value       DOUBLE PRECISION NOT NULL
);
"""

CREATE_LOG_INDEX = """
CREATE INDEX IF NOT EXISTS idx_metrics_log_spacecraft_name_time
ON spacecraft_metrics_log (spacecraft, name, time DESC);
"""

CREATE_CURRENT_VIEW = """
CREATE OR REPLACE VIEW spacecraft_metrics_current AS
SELECT DISTINCT ON (spacecraft, name)
    spacecraft, name, value, time
FROM spacecraft_metrics_log
ORDER BY spacecraft, name, time DESC;
"""

INSERT_ROW = """
INSERT INTO spacecraft_metrics_log (time, spacecraft, name, value)
VALUES ($1, $2, $3, $4)
"""

async def _ensure_schema(conn: asyncpg.Connection):
    await conn.execute(CREATE_LOG)
    await conn.execute(CREATE_LOG_INDEX)
    await conn.execute(CREATE_CURRENT_VIEW)

async def _append_metrics(
    conn: asyncpg.Connection,
    spacecraft_name: str,
    metrics: dict[str, float],
    now: datetime,
):
    rows = [(now, spacecraft_name, name, value) for name, value in metrics.items()]
    await conn.executemany(INSERT_ROW, rows)


# =============================================================================
# MONITOR
# =============================================================================

async def monitor(
    sc: Spacecraft,
    queue: asyncio.Queue,
    dsn: str,
    initial_state: SpacecraftState | None = None,
):
    """
    Async monitor. Single queue, single consumer.
    Runs until cancelled.

    Grafana — current state panel:
      SELECT name, value, time FROM spacecraft_metrics_current
      WHERE spacecraft = 'DemoSat-1'

    Grafana — time series panel:
      SELECT time, value FROM spacecraft_metrics_log
      WHERE name = 'delta_v_m_per_s' AND spacecraft = 'DemoSat-1'
      ORDER BY time
    """
    conn = await asyncpg.connect(dsn)
    await _ensure_schema(conn)
    state = initial_state or SpacecraftState()
    print(f"Monitor running for '{sc.name}'")

    try:
        while True:
            event = await queue.get()
            state = reduce(state, event)
            metrics = evaluate_rules(sc, state)
            await _append_metrics(conn, sc.name, metrics, datetime.now(timezone.utc))
            queue.task_done()
    except asyncio.CancelledError:
        pass
    finally:
        await conn.close()


# =============================================================================
# FAKE PRODUCERS — replace with real stream readers in production
# =============================================================================

async def _fake_position_stream(queue: asyncio.Queue):
    x = 0.0
    while True:
        await queue.put(PositionUpdate(Position(x, 0.0, 400_000.0), datetime.now(timezone.utc)))
        x += 100.0
        await asyncio.sleep(1.0)

async def _fake_command_stream(queue: asyncio.Queue):
    import random
    subs = ["propulsion", "power", "thermal", "avionics"]
    while True:
        await queue.put(CommandUpdate(event=CommandEvent(
            subsystem=random.choice(subs), command="ping",
            accepted=random.random() > 0.1,
            latency_ms=random.uniform(10, 150),
            time=datetime.now(timezone.utc),
        )))
        await asyncio.sleep(0.5)

async def _fake_subsystem_stream(queue: asyncio.Queue):
    import random
    subs = ["propulsion", "power", "thermal", "avionics"]
    while True:
        for s in subs:
            await queue.put(SubsystemUpdate(subsystem=s, availability=random.uniform(0.7, 1.0)))
        await asyncio.sleep(2.0)

async def _fake_propellant_stream(queue: asyncio.Queue, initial_mass: float):
    mass = initial_mass
    while mass > 0:
        mass = max(0.0, mass - 0.1)
        await queue.put(PropellantUpdate(propellant_mass_kg=mass))
        await asyncio.sleep(1.0)

async def main_fake(sc: Spacecraft):
    queue: asyncio.Queue = asyncio.Queue()
    mission = Mission("Survey-1", tasks=(
        Task("init",    ("avionics",),              0),
        Task("burn",    ("propulsion", "avionics"), 1),
        Task("observe", ("power", "avionics"),      2),
    ))
    await queue.put(MissionUpdate(mission=mission, current_task=mission.tasks[1]))
    DSN = "postgresql://user:password@localhost/spacecraft"
    async with asyncio.TaskGroup() as tg:
        tg.create_task(_fake_position_stream(queue))
        tg.create_task(_fake_command_stream(queue))
        tg.create_task(_fake_subsystem_stream(queue))
        tg.create_task(_fake_propellant_stream(queue, sc.propulsion.propellant_mass_kg))
        tg.create_task(monitor(sc, queue, DSN))


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":

    sc = Spacecraft(
        name="DemoSat-1",
        structure=Structure(material="Aluminum honeycomb", dry_mass_kg=35.0),
        propulsion=Propulsion(
            thrust_n=22.0, isp_s=220.0,
            propellant=Propellant("Hydrazine", 1004),
            dry_mass_kg=12.0, propellant_mass_kg=28.0,
        ),
        power=Power(generation_w=180.0, storage_wh=600.0, dry_mass_kg=15.0),
        avionics=Avionics(
            cpu="RAD750", sensors=("IMU", "Star Tracker"),
            average_power_w=35.0, dry_mass_kg=8.0,
        ),
        thermal=ThermalControl(method="Radiators", dry_mass_kg=6.0),
        payloads=(Payload("Optical Camera", dry_mass_kg=12.0, average_power_w=25.0),),
    )

    state = SpacecraftState(
        position=Position(100.0, 0.0, 400_000.0),
        mission=Mission("Survey-1", tasks=(
            Task("init",    ("avionics",),              0),
            Task("burn",    ("propulsion", "avionics"), 1),
            Task("observe", ("power", "avionics"),      2),
        )),
        current_task=Task("burn", ("propulsion", "avionics"), 1),
        subsystems={"propulsion": 1.0, "avionics": 0.9, "power": 0.8, "thermal": 1.0},
        commands=[
            CommandEvent("propulsion", "fire", True,  45.0,  datetime.now(timezone.utc)),
            CommandEvent("propulsion", "fire", False, 120.0, datetime.now(timezone.utc)),
            CommandEvent("avionics",   "ping", True,  12.0,  datetime.now(timezone.utc)),
        ],
    )

    print("=== Static rule evaluation ===")
    for name, value in evaluate_rules(sc, state).items():
        print(f"  {name:<40} : {value}")

    print("\n=== After burn event ===")
    state = reduce(state, PropellantUpdate(propellant_mass_kg=14.0))
    print(f"  delta_v              : {delta_v(sc, state):.1f} m/s")
    print(f"  propellant_remaining : {propellant_remaining_pct(sc, state):.1f} %")
    print(f"  mission_viable       : {mission_viable(sc, state)}")

    # asyncio.run(main_fake(sc))
