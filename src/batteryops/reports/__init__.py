"""UI-facing reporting helpers."""

from batteryops.reports.demo import build_demo_report, build_demo_timeline
from batteryops.reports.incidents import generate_incident_report

__all__ = [
    "build_demo_report",
    "build_demo_timeline",
    "generate_incident_report",
]
