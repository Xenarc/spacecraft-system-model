"""
spacecraft.py

Three concerns, cleanly separated:

  1. System model      — domain dataclasses describing the spacecraft
  2. Fault model       — fault objects with Bayesian priors and detection functions
  3. State loader      — load model state from upstream files (YAML / JSON / CSV)

Nothing is coupled across concerns except the shared dataclass types.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml  # pip install pyyaml


# =============================================================================
# 1. SYSTEM MODEL
# =============================================================================

@dataclass(frozen=True)
class Propellant:
    name: str
    density_kg_per_m3: float


@dataclass
class PropulsionSystem:
    thrust_n: float
    isp_s: float
    propellant: Propellant
    dry_mass_kg: float
    propellant_mass_kg: float

    @property
    def total_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg

    def delta_v_m_per_s(self, g0: float = 9.80665) -> float:
        m0 = self.total_mass_kg
        mf = self.dry_mass_kg
        if mf <= 0 or m0 <= mf:
            return 0.0
        return self.isp_s * g0 * math.log(m0 / mf)


@dataclass
class PowerSystem:
    generation_w: float
    storage_wh: float
    dry_mass_kg: float

    def endurance_hours(self, average_load_w: float) -> float:
        if average_load_w <= 0:
            return float("inf")
        return self.storage_wh / average_load_w


@dataclass
class Avionics:
    cpu: str
    sensors: List[str]
    average_power_w: float
    dry_mass_kg: float


@dataclass
class ThermalControl:
    method: str
    dry_mass_kg: float


@dataclass
class Structure:
    material: str
    dry_mass_kg: float


@dataclass
class Payload:
    name: str
    dry_mass_kg: float
    average_power_w: float = 0.0


@dataclass
class Spacecraft:
    name: str
    structure: Structure
    propulsion: Optional[PropulsionSystem]
    power: PowerSystem
    avionics: Avionics
    thermal: ThermalControl
    payloads: List[Payload] = field(default_factory=list)

    @property
    def dry_mass_kg(self) -> float:
        masses = [
            self.structure.dry_mass_kg,
            self.power.dry_mass_kg,
            self.avionics.dry_mass_kg,
            self.thermal.dry_mass_kg,
            sum(p.dry_mass_kg for p in self.payloads),
        ]
        if self.propulsion:
            masses.append(self.propulsion.dry_mass_kg)
        return sum(masses)

    @property
    def wet_mass_kg(self) -> float:
        if self.propulsion:
            return self.dry_mass_kg + self.propulsion.propellant_mass_kg
        return self.dry_mass_kg

    @property
    def average_power_w(self) -> float:
        return self.avionics.average_power_w + sum(p.average_power_w for p in self.payloads)

    def delta_v_m_per_s(self) -> float:
        if not self.propulsion:
            return 0.0
        m0 = self.wet_mass_kg
        mf = self.dry_mass_kg
        if mf <= 0 or m0 <= mf:
            return 0.0
        return self.propulsion.isp_s * 9.80665 * math.log(m0 / mf)

    def power_margin(self) -> float:
        return self.power.generation_w - self.average_power_w


# =============================================================================
# 2. FAULT MODEL
# =============================================================================
#
# A Fault is not an event. It is a latent condition with:
#   - a prior probability of being active
#   - a detection function that inspects system state and returns a likelihood
#   - a posterior that is updated via Bayes: P(fault|obs) ∝ P(obs|fault) * P(fault)
#
# The detection function is the sketch — it can be as simple or rich as needed.
# The Bayesian update keeps the math honest without prescribing a framework.

DetectionFn = Callable[[Dict[str, Any]], float]
# A detection function takes a dict of observable telemetry and returns
# P(observation | fault is active), a likelihood in [0, 1].


@dataclass
class Fault:
    """
    A single named fault with a prior probability and a detection function.

    Attributes
    ----------
    name        : human-readable identifier
    subsystem   : which subsystem this fault belongs to
    prior       : P(fault active) before any evidence, in (0, 1)
    detect      : callable(telemetry) -> likelihood P(obs | fault)
    description : optional narrative for the failure narrative generator
    posterior   : updated by observe(); starts equal to prior
    """
    name: str
    subsystem: str
    prior: float
    detect: DetectionFn
    description: str = ""
    posterior: float = field(init=False)

    def __post_init__(self):
        if not 0.0 < self.prior < 1.0:
            raise ValueError(f"Prior for '{self.name}' must be in (0, 1), got {self.prior}")
        self.posterior = self.prior

    def observe(self, telemetry: Dict[str, Any]) -> float:
        """
        Bayesian update given a telemetry snapshot.

        P(fault | obs) = P(obs | fault) * P(fault)
                         --------------------------------
                         P(obs | fault)*P(fault) + P(obs | ~fault)*(1-P(fault))

        P(obs | ~fault) is approximated as (1 - likelihood) for a binary sensor model.
        This is the simplest useful form; swap in a full likelihood table when available.
        """
        likelihood_given_fault = self.detect(telemetry)
        likelihood_given_nominal = 1.0 - likelihood_given_fault  # crude complement

        numerator = likelihood_given_fault * self.posterior
        denominator = numerator + likelihood_given_nominal * (1.0 - self.posterior)

        if denominator == 0.0:
            return self.posterior  # no information

        self.posterior = numerator / denominator
        return self.posterior

    def reset(self):
        """Reset posterior to prior (start of a new diagnostic window)."""
        self.posterior = self.prior


# ---- Detection function sketches ----
#
# These are intentionally simple. The pattern is:
#   1. extract the relevant telemetry key(s)
#   2. apply a threshold or soft sigmoid
#   3. return a likelihood in [0, 1]
#
# Replace the threshold logic with a trained classifier, lookup table,
# or physics-based model as fidelity demands.

def detect_battery_overheat(telemetry: Dict[str, Any]) -> float:
    """High battery temperature is evidence of a stuck heater fault."""
    temp = telemetry.get("battery_temp_c", 20.0)
    # Nominal range: 0–35 °C. Above 45 °C is suspicious. Above 60 °C is strong evidence.
    if temp < 35.0:
        return 0.02
    if temp > 60.0:
        return 0.95
    # Linear ramp in the middle
    return 0.02 + (temp - 35.0) / 25.0 * 0.93


def detect_bus_undervoltage(telemetry: Dict[str, Any]) -> float:
    """Low bus voltage is evidence of a power regulation fault."""
    voltage = telemetry.get("bus_voltage_v", 28.0)
    nominal = 28.0
    if voltage >= nominal * 0.95:
        return 0.01
    if voltage < nominal * 0.80:
        return 0.90
    # Soft ramp between 80–95% of nominal
    drop_fraction = (nominal - voltage) / (nominal * 0.15)
    return 0.01 + drop_fraction * 0.89


def detect_thruster_blockage(telemetry: Dict[str, Any]) -> float:
    """Commanded thrust not producing expected acceleration is evidence of blockage."""
    commanded = telemetry.get("thrust_commanded_n", 0.0)
    measured_accel = telemetry.get("accel_measured_m_per_s2", 0.0)
    mass_estimate = telemetry.get("mass_estimate_kg", 100.0)

    if commanded <= 0.0:
        return 0.01  # not commanded, no evidence either way

    expected_accel = commanded / mass_estimate
    if expected_accel <= 0.0:
        return 0.01

    ratio = measured_accel / expected_accel  # 1.0 = nominal, 0.0 = full blockage
    ratio = max(0.0, min(ratio, 1.0))
    return 1.0 - ratio  # low delivery = high fault likelihood


def detect_sensor_dropout(telemetry: Dict[str, Any]) -> float:
    """Missing or stale sensor readings indicate a sensor dropout fault."""
    stale_count = telemetry.get("stale_sensor_count", 0)
    total_sensors = telemetry.get("total_sensor_count", 10)
    if total_sensors == 0:
        return 0.5
    fraction_stale = stale_count / total_sensors
    # Any staleness above 10% is notable
    return min(fraction_stale * 2.5, 1.0)


# ---- Fault catalog ----
#
# A FaultCatalog is just a named collection. The interesting operation is
# batch_observe: update all posteriors from a single telemetry snapshot,
# then rank faults by posterior for diagnostics.

@dataclass
class FaultCatalog:
    faults: List[Fault] = field(default_factory=list)

    def add(self, fault: Fault) -> None:
        self.faults.append(fault)

    def batch_observe(self, telemetry: Dict[str, Any]) -> List[tuple[str, float]]:
        """
        Update all fault posteriors from a telemetry snapshot.
        Returns a list of (fault_name, posterior) sorted by posterior descending.
        """
        results = []
        for fault in self.faults:
            posterior = fault.observe(telemetry)
            results.append((fault.name, posterior))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def top(self, n: int = 3) -> List[Fault]:
        """Return the n most probable active faults."""
        return sorted(self.faults, key=lambda f: f.posterior, reverse=True)[:n]

    def reset_all(self) -> None:
        for fault in self.faults:
            fault.reset()


def default_fault_catalog() -> FaultCatalog:
    catalog = FaultCatalog()
    catalog.add(Fault(
        name="heater_stuck_on",
        subsystem="thermal",
        prior=0.01,
        detect=detect_battery_overheat,
        description="Heater relay failed closed; battery temperature rising.",
    ))
    catalog.add(Fault(
        name="power_reg_fault",
        subsystem="power",
        prior=0.005,
        detect=detect_bus_undervoltage,
        description="Bus voltage regulator degraded; downstream brownout risk.",
    ))
    catalog.add(Fault(
        name="thruster_blockage",
        subsystem="propulsion",
        prior=0.008,
        detect=detect_thruster_blockage,
        description="Thruster feed line partially blocked; thrust delivery reduced.",
    ))
    catalog.add(Fault(
        name="sensor_dropout",
        subsystem="avionics",
        prior=0.015,
        detect=detect_sensor_dropout,
        description="One or more sensors returning stale or missing data.",
    ))
    return catalog


# =============================================================================
# 3. STATE LOADER
# =============================================================================
#
# Load spacecraft model state from upstream files.
# Supports three formats:
#   - YAML  (structured config, human-authored)
#   - JSON  (API output, telemetry archive)
#   - CSV   (parameter tables, mass budgets from spreadsheets)
#
# The loader is intentionally narrow: it builds the dataclass tree from raw
# dicts. No magic introspection — explicit field mapping keeps it debuggable.

class SpacecraftLoader:
    """
    Load a Spacecraft (and optionally a FaultCatalog) from upstream files.

    Supported formats
    -----------------
    YAML : full spacecraft config (preferred for human-authored files)
    JSON : same schema as YAML, useful for API-sourced configs
    CSV  : mass budget table; partial load — populates component masses only

    Usage
    -----
        sc = SpacecraftLoader.from_yaml("mission_config.yaml")
        sc = SpacecraftLoader.from_json("demosat.json")
        sc = SpacecraftLoader.from_mass_budget_csv("mass_budget.csv")
    """

    # ---- YAML / JSON (full model) ----------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Spacecraft:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> Spacecraft:
        with open(path, "r") as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> Spacecraft:
        """
        Build a Spacecraft from a raw dict.
        Expected top-level keys:
          name, structure, propulsion (optional), power, avionics, thermal, payloads
        """
        propulsion = None
        if "propulsion" in d and d["propulsion"] is not None:
            p = d["propulsion"]
            propellant = Propellant(**p["propellant"])
            propulsion = PropulsionSystem(
                thrust_n=p["thrust_n"],
                isp_s=p["isp_s"],
                propellant=propellant,
                dry_mass_kg=p["dry_mass_kg"],
                propellant_mass_kg=p["propellant_mass_kg"],
            )

        av = d["avionics"]
        avionics = Avionics(
            cpu=av["cpu"],
            sensors=av["sensors"],
            average_power_w=av["average_power_w"],
            dry_mass_kg=av["dry_mass_kg"],
        )

        payloads = [
            Payload(**pl) for pl in d.get("payloads", [])
        ]

        return Spacecraft(
            name=d["name"],
            structure=Structure(**d["structure"]),
            propulsion=propulsion,
            power=PowerSystem(**d["power"]),
            avionics=avionics,
            thermal=ThermalControl(**d["thermal"]),
            payloads=payloads,
        )

    # ---- CSV (partial: mass budget table) -----------------------------------
    #
    # Expected columns: subsystem, component, dry_mass_kg
    # Rows map to subsystem buckets; missing subsystems use defaults.
    # This is a sketch — real mass budgets vary in structure, so treat
    # this as a starting point to adapt to your actual spreadsheet format.

    @classmethod
    def from_mass_budget_csv(
        cls,
        path: str | Path,
        spacecraft_name: str = "unnamed",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Spacecraft:
        """
        Build a Spacecraft from a CSV mass budget.

        CSV format (header row required):
            subsystem,component,dry_mass_kg

        Example rows:
            structure,primary_structure,35.0
            power,solar_array,15.0
            avionics,flight_computer,8.0
            thermal,radiator_panel,6.0
            propulsion,prop_system_dry,12.0
            payload,optical_camera,12.0

        Non-mass fields (thrust, ISP, etc.) must be supplied via `defaults`.
        """
        masses: Dict[str, float] = {}

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subsystem = row["subsystem"].strip().lower()
                mass = float(row["dry_mass_kg"])
                masses[subsystem] = masses.get(subsystem, 0.0) + mass

        d = defaults or {}

        # Structure
        structure = Structure(
            material=d.get("structure_material", "unspecified"),
            dry_mass_kg=masses.get("structure", d.get("structure_dry_mass_kg", 0.0)),
        )

        # Power
        power = PowerSystem(
            generation_w=d.get("power_generation_w", 0.0),
            storage_wh=d.get("power_storage_wh", 0.0),
            dry_mass_kg=masses.get("power", d.get("power_dry_mass_kg", 0.0)),
        )

        # Avionics
        avionics = Avionics(
            cpu=d.get("avionics_cpu", "unspecified"),
            sensors=d.get("avionics_sensors", []),
            average_power_w=d.get("avionics_power_w", 0.0),
            dry_mass_kg=masses.get("avionics", d.get("avionics_dry_mass_kg", 0.0)),
        )

        # Thermal
        thermal = ThermalControl(
            method=d.get("thermal_method", "unspecified"),
            dry_mass_kg=masses.get("thermal", d.get("thermal_dry_mass_kg", 0.0)),
        )

        # Propulsion (optional)
        propulsion = None
        if "propulsion" in masses or "propellant" in masses:
            propellant = Propellant(
                name=d.get("propellant_name", "unspecified"),
                density_kg_per_m3=d.get("propellant_density_kg_per_m3", 1000.0),
            )
            propulsion = PropulsionSystem(
                thrust_n=d.get("thrust_n", 0.0),
                isp_s=d.get("isp_s", 0.0),
                propellant=propellant,
                dry_mass_kg=masses.get("propulsion", 0.0),
                propellant_mass_kg=masses.get("propellant", d.get("propellant_mass_kg", 0.0)),
            )

        # Payloads — each CSV row with subsystem="payload" becomes one Payload
        payloads = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["subsystem"].strip().lower() == "payload":
                    payloads.append(Payload(
                        name=row.get("component", "unnamed_payload"),
                        dry_mass_kg=float(row["dry_mass_kg"]),
                        average_power_w=float(row.get("average_power_w", 0.0)),
                    ))

        return Spacecraft(
            name=spacecraft_name,
            structure=structure,
            propulsion=propulsion,
            power=power,
            avionics=avionics,
            thermal=thermal,
            payloads=payloads,
        )


# =============================================================================
# EXAMPLE — wire everything together
# =============================================================================

if __name__ == "__main__":

    # --- Build inline (no file) ---
    hydrazine = Propellant(name="Hydrazine", density_kg_per_m3=1004)

    sc = Spacecraft(
        name="DemoSat-1",
        structure=Structure(material="Aluminum honeycomb", dry_mass_kg=35.0),
        propulsion=PropulsionSystem(
            thrust_n=22.0,
            isp_s=220.0,
            propellant=hydrazine,
            dry_mass_kg=12.0,
            propellant_mass_kg=28.0,
        ),
        power=PowerSystem(generation_w=180.0, storage_wh=600.0, dry_mass_kg=15.0),
        avionics=Avionics(
            cpu="RAD750",
            sensors=["IMU", "Star Tracker"],
            average_power_w=35.0,
            dry_mass_kg=8.0,
        ),
        thermal=ThermalControl(method="Radiators", dry_mass_kg=6.0),
        payloads=[Payload(name="Optical Camera", dry_mass_kg=12.0, average_power_w=25.0)],
    )

    print("=== System Model ===")
    print(f"  Dry mass  : {sc.dry_mass_kg} kg")
    print(f"  Wet mass  : {sc.wet_mass_kg} kg")
    print(f"  Δv        : {sc.delta_v_m_per_s():.1f} m/s")
    print(f"  Power margin: {sc.power_margin():.1f} W")

    # --- Fault model: batch observe against a synthetic telemetry snapshot ---
    catalog = default_fault_catalog()

    telemetry = {
        "battery_temp_c": 52.0,       # warm — somewhat suspicious
        "bus_voltage_v": 25.5,        # sagging
        "thrust_commanded_n": 22.0,
        "accel_measured_m_per_s2": 0.18,  # expected ~0.22, slightly low
        "mass_estimate_kg": sc.wet_mass_kg,
        "stale_sensor_count": 1,
        "total_sensor_count": len(sc.avionics.sensors),
    }

    print("\n=== Fault Posteriors (after one observation) ===")
    ranked = catalog.batch_observe(telemetry)
    for name, posterior in ranked:
        print(f"  {name:<25} P={posterior:.4f}")

    print("\n=== Top Fault ===")
    top = catalog.top(1)[0]
    print(f"  {top.name}: {top.description}")

    # --- Loader: show how you'd load from YAML (file not present, just schema) ---
    print("\n=== Loader (YAML schema reference) ===")
    schema = """
    # mission_config.yaml — expected structure for SpacecraftLoader.from_yaml()
    #
    # name: DemoSat-1
    # structure:
    #   material: Aluminum honeycomb
    #   dry_mass_kg: 35.0
    # propulsion:
    #   thrust_n: 22.0
    #   isp_s: 220.0
    #   dry_mass_kg: 12.0
    #   propellant_mass_kg: 28.0
    #   propellant:
    #     name: Hydrazine
    #     density_kg_per_m3: 1004
    # power:
    #   generation_w: 180.0
    #   storage_wh: 600.0
    #   dry_mass_kg: 15.0
    # avionics:
    #   cpu: RAD750
    #   sensors: [IMU, Star Tracker]
    #   average_power_w: 35.0
    #   dry_mass_kg: 8.0
    # thermal:
    #   method: Radiators
    #   dry_mass_kg: 6.0
    # payloads:
    #   - name: Optical Camera
    #     dry_mass_kg: 12.0
    #     average_power_w: 25.0
    """
    print(schema)