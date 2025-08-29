import numpy as np
import pandas as pd
import simpy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque

from capacity_generator import CataractCapacityGenerator
from patient_generator import CataractPatientGenerator
from rtt_generator import RTTWaitTimeGenerator
from service_time_generator import CataractServiceTimeGenerator
from arrival_generator import HealthcareArrivalGenerator
from provider import CataractProviderGenerator



class TriagePolicy:
    def assign_provider(self, patient: "Patient", date: pd.Timestamp, sim: "CataractSimulationSimPy"):
        return None


class NoopPolicy(TriagePolicy):
    pass


class AntiBreachISPolicy(TriagePolicy):
    def __init__(self, is_provider_priority: Optional[List[str]] = None, days_to_breach_threshold: int = 21):
        self.is_provider_priority = is_provider_priority or ["SPAMEDICA", "NEWMEDICA", "FITZWILLIAM"]
        self.days_to_breach_threshold = days_to_breach_threshold

    def assign_provider(self, patient, date, sim):
        if patient.days_until_target(date) > self.days_to_breach_threshold:
            return None
        if patient.provider_type == "Independent":
            return None
        for provider in self.is_provider_priority:
            if provider in sim._known_providers:
                return provider
        return None



@dataclass
class Patient:
    patient_id: str
    referral_date: pd.Timestamp
    surgery_target_date: pd.Timestamp
    priority: int
    priority_name: str
    hrg_code: str
    hrg_description: str
    complexity_level: int
    complexity_category: str

    surgery_duration: float
    preop_assessment: float
    postop_1week: float
    postop_6week: float
    theatre_setup: float
    theatre_changeover: float
    total_theatre_time: float
    total_episode_time: float
    base_surgery_duration_minutes: int

    assessment_provider: str
    surgery_provider: str
    is_independent_sector: bool
    provider_loyalty: bool
    provider_type: str
    estimated_daily_capacity: float

    is_bilateral: bool = False
    inter_eye_interval: Optional[int] = None
    bilateral_pair: Optional[str] = None
    is_second_eye: bool = False

    actual_surgery_date: Optional[pd.Timestamp] = None
    actual_wait_days: Optional[int] = None
    served_bucket: Optional[str] = None
    completed: bool = False
    cancelled: bool = False

    def days_waiting(self, current_date: pd.Timestamp) -> int:
        return (current_date - self.referral_date).days

    def days_until_target(self, current_date: pd.Timestamp) -> int:
        return (self.surgery_target_date - current_date).days

    def is_overdue(self, current_date: pd.Timestamp) -> bool:
        return current_date >= self.surgery_target_date

    def within_18_weeks(self) -> bool:
        if self.actual_wait_days is not None:
            return self.actual_wait_days <= 126
        return (self.surgery_target_date - self.referral_date).days <= 126


# ------------------------- Simulation -------------------------

class CataractSimulationSimPy:
    def __init__(self, start_date="2024-01-01", random_seed=42):
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.env = simpy.Environment()

        self.start_date = pd.Timestamp(start_date)
        self._init_generators()

        # scheduling knobs
        self.min_lead_time_days = 14
        self.early_pull_window_days = 28
        self.early_pull_fraction = 0.15

        # capacity realism
        self.capacity_disruption_mean = 0.92
        self.capacity_disruption_sd = 0.025

        # mix across buckets
        self.frac_at_risk = 0.25
        self.frac_due = 0.35
        self.wait_threshold_days = 126
        self.compliance_target = 0.69

        # state
        self.patients: List[Patient] = []
        self.completed_patients: List[Patient] = []
        self.daily_stats: List[Dict] = []
        self.patient_counter = 0
        self.daily_capacity: Dict[pd.Timestamp, int] = {}
        self.daily_arrivals_schedule: Dict[pd.Timestamp, int] = {}
        self.measurement_start: Optional[pd.Timestamp] = None

        # providers
        self.provider_stats = defaultdict(lambda: {
            "total_assessments": 0,
            "total_surgeries": 0,
            "total_patients": 0,
            "avg_wait_days": 0,
            "rtt_compliance": 0,
            "is_independent": False,
            "estimated_capacity": 0,
        })
        self._known_providers: set[str] = set()
        self._provider_cap_estimate: Dict[str, float] = defaultdict(float)
        self._provider_cap_count: Dict[str, int] = defaultdict(int)

        # queues and policy
        self.waiting_queue = deque()
        self.policy: TriagePolicy = NoopPolicy()

        self._setup_yearly_capacity(2024)

    # --- public ---
    def set_policy(self, policy: TriagePolicy):
        self.policy = policy or NoopPolicy()

    # --- generators & capacity ---
    def _init_generators(self):
        self.capacity_gen = CataractCapacityGenerator()
        self.patient_gen = CataractPatientGenerator(random_seed=self.random_seed)
        self.rtt_gen = RTTWaitTimeGenerator()
        self.service_gen = CataractServiceTimeGenerator()
        self.arrival_gen = HealthcareArrivalGenerator()
        self.arrival_gen.set_random_seed(self.random_seed)
        self.provider_gen = CataractProviderGenerator()

    def _setup_yearly_capacity(self, year: int):
        total_capacity = 0
        capacity_days = 0
        for month in range(1, 13):
            monthly_schedule = self.capacity_gen.generate_monthly_schedule(month, year)
            for _, row in monthly_schedule.iterrows():
                date_key = row["date"].date()
                base_capacity = row["capacity"]
                realistic_capacity = max(0, int(base_capacity * 0.80))
                self.daily_capacity[date_key] = realistic_capacity
                if realistic_capacity > 0:
                    capacity_days += 1
                    total_capacity += realistic_capacity
        avg_cap = total_capacity / capacity_days if capacity_days else 0
        print(f"Capacity: {capacity_days} operating days, avg {avg_cap:.1f}/day")

    def setup_arrival_schedule(self, duration_days: int, backlog_days: int, method: str, warmup_days: int):
        total_days = backlog_days + warmup_days + duration_days
        schedule_start = self.start_date - pd.Timedelta(days=(backlog_days + warmup_days))
        arrival_ts = self.arrival_gen.generate_arrivals_time_series(
            start_date=schedule_start.strftime("%Y-%m-%d"),
            num_days=total_days,
            method=method,
        )
        self.daily_arrivals_schedule = {}
        if len(arrival_ts) > 0:
            daily_counts = arrival_ts.groupby("date").size().to_dict()
            current = schedule_start
            for _ in range(total_days):
                self.daily_arrivals_schedule[current.date()] = int(daily_counts.get(current.date(), 0))
                current += pd.Timedelta(days=1)
        else:
            current = schedule_start
            for _ in range(total_days):
                self.daily_arrivals_schedule[current.date()] = int(
                    self.arrival_gen.get_expected_daily_arrivals(current.weekday())
                )
                current += pd.Timedelta(days=1)
        total_arrivals = sum(self.daily_arrivals_schedule.values())
        print(f"Arrivals: {total_arrivals:,} over {total_days} days (avg {total_arrivals/total_days:.1f}/day)")

    # --- patient creation & provider stats ---
    def _create_patient(self, referral_date: pd.Timestamp) -> Patient:
        self.patient_counter += 1
        patient_id = f"P{self.patient_counter:06d}"
        pdata = self.patient_gen.generate_patient()
        provider_pathway = self.provider_gen.generate_complete_pathway()
        is_bilateral = np.random.random() < 0.398
        inter_eye_interval = max(30, int(np.random.lognormal(3.7, 1.2))) if is_bilateral else None

        rtt_wait = self.rtt_gen.generate_rtt_wait(
            priority=pdata["priority"],
            hrg=pdata["hrg_code"],
            referral_month=referral_date.month,
        )
        surgery_target_date = referral_date + pd.Timedelta(days=rtt_wait)

        etimes = self.service_gen.generate_complete_episode_times(
            hrg_code=pdata["hrg_code"],
            complexity=pdata["complexity_category"].lower(),
            activity_type="Day Case",
        )

        patient = Patient(
            patient_id=patient_id,
            referral_date=referral_date,
            surgery_target_date=surgery_target_date,
            priority=pdata["priority"],
            priority_name=pdata["priority_name"],
            hrg_code=pdata["hrg_code"],
            hrg_description=pdata["hrg_description"],
            complexity_level=pdata["complexity_level"],
            complexity_category=pdata["complexity_category"],
            surgery_duration=etimes["surgery_duration"],
            preop_assessment=etimes["preop_assessment"],
            postop_1week=etimes["postop_1week"],
            postop_6week=etimes["postop_6week"],
            theatre_setup=etimes["theatre_setup"],
            theatre_changeover=etimes["theatre_changeover"],
            total_theatre_time=etimes["total_theatre_time"],
            total_episode_time=etimes["total_episode_time"],
            base_surgery_duration_minutes=pdata["base_surgery_duration_minutes"],
            assessment_provider=provider_pathway["assessment_provider"],
            surgery_provider=provider_pathway["surgery_provider"],
            is_independent_sector=provider_pathway["is_independent_sector"],
            provider_loyalty=provider_pathway["provider_loyalty"],
            provider_type=provider_pathway["provider_type"],
            estimated_daily_capacity=provider_pathway["estimated_daily_capacity"],
            is_bilateral=is_bilateral,
            inter_eye_interval=inter_eye_interval,
        )
        self._update_provider_stats(patient)
        return patient

    def _update_provider_stats(self, patient: Patient):
        self.provider_stats[patient.assessment_provider]["total_assessments"] += 1
        self.provider_stats[patient.assessment_provider]["is_independent"] = patient.is_independent_sector

        self.provider_stats[patient.surgery_provider]["total_surgeries"] += 1
        self.provider_stats[patient.surgery_provider]["total_patients"] += 1
        self.provider_stats[patient.surgery_provider]["is_independent"] = patient.is_independent_sector

        self._known_providers.add(patient.surgery_provider)
        prior = self._provider_cap_estimate[patient.surgery_provider]
        n = self._provider_cap_count[patient.surgery_provider]
        new_avg = (prior * n + max(1.0, float(patient.estimated_daily_capacity))) / (n + 1)
        self._provider_cap_estimate[patient.surgery_provider] = new_avg
        self._provider_cap_count[patient.surgery_provider] = n + 1
        self.provider_stats[patient.surgery_provider]["estimated_capacity"] = new_avg

    # --- processes ---
    def patient_arrival_process(self, backlog_days: int, warmup_days: int, duration_days: int):
        backlog_start = -(backlog_days + warmup_days)
        backlog_count = 0
        for day_offset in range(backlog_start, -warmup_days):
            arrival_date = self.start_date + pd.Timedelta(days=day_offset)
            daily_arrivals = self.daily_arrivals_schedule.get(
                arrival_date.date(), self.arrival_gen.get_expected_daily_arrivals(arrival_date.weekday())
            )
            for _ in range(daily_arrivals):
                p = self._create_patient(arrival_date)
                self.patients.append(p)
                self.waiting_queue.append(p)
                backlog_count += 1

        print(f"Backlog patients: {backlog_count}")

        for day in range(warmup_days + duration_days):
            arrival_date = self.start_date + pd.Timedelta(days=day - warmup_days)
            daily_arrivals = self.daily_arrivals_schedule.get(
                arrival_date.date(), self.arrival_gen.get_expected_daily_arrivals(arrival_date.weekday())
            )
            for _ in range(daily_arrivals):
                p = self._create_patient(arrival_date)
                self.patients.append(p)
                self.waiting_queue.append(p)
            yield self.env.timeout(1)

    def surgery_scheduling_process(self, warmup_days: int):
        self.measurement_start = self.start_date
        while True:
            current_day = self.env.now
            current_date = self.start_date + pd.Timedelta(days=current_day - warmup_days)
            base_capacity = self.daily_capacity.get(current_date.date(), 0)

            if base_capacity > 0:
                eligible = self._get_eligible_patients(current_date)

                # policy routing
                for p in eligible:
                    new_provider = self.policy.assign_provider(p, current_date, self)
                    if new_provider and new_provider != p.surgery_provider:
                        p.surgery_provider = new_provider
                        if new_provider.isalpha() and new_provider.upper() in {"SPAMEDICA", "NEWMEDICA", "FITZWILLIAM"}:
                            p.provider_type = "Independent"
                            p.is_independent_sector = True

                due_start = sum(1 for p in eligible if p.is_overdue(current_date))
                stats = self._provider_aware_schedule(current_date, base_capacity, eligible, due_start)
                stats["arrivals"] = self.daily_arrivals_schedule.get(current_date.date(), 0)
                self.daily_stats.append(stats)

            yield self.env.timeout(1)

    # --- scheduling helpers ---
    def _effective_capacity(self, base_capacity: int) -> int:
        if base_capacity <= 0:
            return 0
        factor = np.clip(
            np.random.normal(self.capacity_disruption_mean, self.capacity_disruption_sd), 0.75, 1.00
        )
        return max(1, int(np.floor(base_capacity * factor)))

    def _triage_score(self, p: Patient, date: pd.Timestamp) -> float:
        days_to_breach = p.days_until_target(date)
        return 5.0 * p.priority - 0.5 * days_to_breach + 0.2 * p.complexity_level + (2.0 if p.is_bilateral else 0.0)

    def _get_eligible_patients(self, date: pd.Timestamp) -> List[Patient]:
        min_lead = pd.Timedelta(days=self.min_lead_time_days)
        return [
            patient for patient in list(self.waiting_queue)
            if (not patient.completed and not patient.cancelled and date >= patient.referral_date + min_lead)
        ]

    def _categorize_patients(self, eligible: List[Patient], date: pd.Timestamp) -> Tuple[List, List, List, List]:
        due, at_risk, near_due, fifo_early = [], [], [], []
        taken = set()

        for p in eligible:
            if p.is_overdue(date):
                due.append(p); taken.add(p.patient_id)

        for p in eligible:
            if p.patient_id in taken:
                continue
            waited = p.days_waiting(date)
            if 115 <= waited < 126:
                at_risk.append(p); taken.add(p.patient_id)

        for p in eligible:
            if p.patient_id in taken:
                continue
            if 0 <= p.days_until_target(date) <= self.early_pull_window_days:
                near_due.append(p); taken.add(p.patient_id)

        for p in eligible:
            if p.patient_id not in taken:
                fifo_early.append(p)

        def key_fn(x: Patient):
            return (-x.priority, x.referral_date, -self._triage_score(x, date), x.complexity_level)

        due.sort(key=key_fn)
        at_risk.sort(key=key_fn)
        near_due.sort(key=key_fn)
        fifo_early.sort(key=key_fn)
        return due, at_risk, near_due, fifo_early

    def _distribute_capacity_across_providers(self, date: pd.Timestamp, effective_capacity: int, providers: List[str]) -> Dict[str, int]:
        if effective_capacity <= 0 or not providers:
            return {p: 0 for p in providers}

        raw = {p: max(1.0, self._provider_cap_estimate.get(p, 1.0)) for p in providers}
        total_raw = float(sum(raw.values()))
        if total_raw <= 0:
            equal = max(0, effective_capacity // len(providers))
            caps = {p: equal for p in providers}
        else:
            caps = {p: int(np.floor(effective_capacity * (raw[p] / total_raw))) for p in providers}

        assigned = sum(caps.values())
        leftover = effective_capacity - assigned
        if leftover > 0:
            for p in sorted(providers, key=lambda x: raw[x], reverse=True):
                if leftover <= 0:
                    break
                caps[p] += 1
                leftover -= 1
        return caps

    def _provider_aware_schedule(self, date: pd.Timestamp, base_capacity: int, eligible_patients: List[Patient], due_start_count: int) -> Dict:
        effective_capacity = self._effective_capacity(base_capacity)
        if effective_capacity == 0:
            return {
                "date": date, "capacity": 0, "effective_capacity": 0,
                "patients_ready": 0, "patients_completed": 0, "patients_cancelled": 0,
                "total_surgery_time": 0, "total_theatre_time": 0,
                "utilization": 0.0, "avg_surgery_time": 0.0,
                "queue_overflow": 0, "due_start": 0, "due_scheduled": 0, "due_end": 0,
                "complexity_distribution": {}, "provider_distribution": {}
            }

        by_provider: Dict[str, List[Patient]] = defaultdict(list)
        providers_today = set()
        for p in eligible_patients:
            provider = p.surgery_provider or "UNKNOWN"
            providers_today.add(provider)
            by_provider[provider].append(p)
        providers_today = list(providers_today)

        per_provider_caps = self._distribute_capacity_across_providers(date, effective_capacity, providers_today)

        completed_count = 0
        total_surgery_time = 0.0
        total_theatre_time = 0.0
        complexity_completed: Dict[int, int] = {}
        provider_completed: Dict[str, int] = {}
        due_scheduled = 0

        for provider in providers_today:
            cap = per_provider_caps.get(provider, 0)
            if cap <= 0:
                continue
            due, at_risk, near_due, fifo_early = self._categorize_patients(by_provider[provider], date)
            scheduled = self._select_patients_for_surgery(due, at_risk, near_due, fifo_early, cap)

            for patient in scheduled:
                if patient.is_overdue(date):
                    due_scheduled += 1
                self._complete_surgery(patient, date)
                completed_count += 1
                total_surgery_time += patient.surgery_duration
                total_theatre_time += patient.total_theatre_time
                complexity_completed[patient.complexity_level] = complexity_completed.get(patient.complexity_level, 0) + 1
                provider_short = (patient.surgery_provider or "UNK").split()[0]
                provider_completed[provider_short] = provider_completed.get(provider_short, 0) + 1

                if patient.is_bilateral and patient.inter_eye_interval:
                    second_eye = self._schedule_bilateral_surgery(patient)
                    if second_eye:
                        self.patients.append(second_eye)
                        self.waiting_queue.append(second_eye)

        if completed_count < effective_capacity:
            remaining_cap = effective_capacity - completed_count
            remaining_candidates = [p for p in eligible_patients if not p.completed and not p.cancelled]
            if remaining_candidates:
                g_due, g_at_risk, g_near, g_fifo = self._categorize_patients(remaining_candidates, date)
                extra = self._select_patients_for_surgery(g_due, g_at_risk, g_near, g_fifo, remaining_cap)
                for patient in extra:
                    if patient.is_overdue(date):
                        due_scheduled += 1
                    self._complete_surgery(patient, date)
                    completed_count += 1
                    total_surgery_time += patient.surgery_duration
                    total_theatre_time += patient.total_theatre_time
                    complexity_completed[patient.complexity_level] = complexity_completed.get(patient.complexity_level, 0) + 1
                    provider_short = (patient.surgery_provider or "UNK").split()[0]
                    provider_completed[provider_short] = provider_completed.get(provider_short, 0) + 1

                    if patient.is_bilateral and patient.inter_eye_interval:
                        second_eye = self._schedule_bilateral_surgery(patient)
                        if second_eye:
                            self.patients.append(second_eye)
                            self.waiting_queue.append(second_eye)

        utilization = min(100.0, (total_theatre_time / (effective_capacity * 60.0)) * 100.0)
        avg_surgery_time = total_surgery_time / completed_count if completed_count > 0 else 0.0
        due_end = max(0, due_start_count - due_scheduled)

        return {
            "date": date,
            "capacity": base_capacity,
            "effective_capacity": effective_capacity,
            "patients_ready": len(eligible_patients),
            "patients_completed": completed_count,
            "patients_cancelled": 0,
            "total_surgery_time": total_surgery_time,
            "total_theatre_time": total_theatre_time,
            "utilization": utilization,
            "avg_surgery_time": avg_surgery_time,
            "queue_overflow": due_end,
            "due_start": due_start_count,
            "due_scheduled": due_scheduled,
            "due_end": due_end,
            "complexity_distribution": complexity_completed,
            "provider_distribution": provider_completed,
        }

    def _select_patients_for_surgery(
        self,
        due: List[Patient],
        at_risk: List[Patient],
        near_due: List[Patient],
        fifo_early: List[Patient],
        effective_capacity: int,
    ) -> List[Patient]:
        scheduled: List[Patient] = []

        cap_due = min(int(round(self.frac_due * effective_capacity)), effective_capacity)
        take = min(len(due), cap_due)
        for i in range(take):
            due[i].served_bucket = "due"
            scheduled.append(due[i])

        remaining = effective_capacity - len(scheduled)
        if remaining > 0:
            cap_risk = min(int(round(self.frac_at_risk * effective_capacity)), remaining)
            take = min(len(at_risk), cap_risk)
            for i in range(take):
                at_risk[i].served_bucket = "at_risk"
                scheduled.append(at_risk[i])

        remaining = effective_capacity - len(scheduled)
        if remaining > 0 and near_due:
            quota = min(int(self.early_pull_fraction * effective_capacity), remaining, len(near_due))
            for i in range(quota):
                near_due[i].served_bucket = "near_due"
                scheduled.append(near_due[i])

        remaining = effective_capacity - len(scheduled)
        if remaining > 0 and fifo_early:
            take = min(remaining, len(fifo_early))
            for i in range(take):
                fifo_early[i].served_bucket = "fifo"
                scheduled.append(fifo_early[i])

        return scheduled

    def _complete_surgery(self, patient: Patient, date: pd.Timestamp):
        patient.completed = True
        patient.actual_surgery_date = date
        patient.actual_wait_days = patient.days_waiting(date)
        if patient in self.waiting_queue:
            self.waiting_queue.remove(patient)
        self.completed_patients.append(patient)

        if patient.surgery_provider in self.provider_stats:
            stats = self.provider_stats[patient.surgery_provider]
            current_avg = stats["avg_wait_days"]
            current_count = stats["total_patients"]
            if current_count > 0:
                stats["avg_wait_days"] = ((current_avg * (current_count - 1)) + patient.actual_wait_days) / current_count
            else:
                stats["avg_wait_days"] = patient.actual_wait_days

            within_18 = 1 if patient.within_18_weeks() else 0
            current_compliance = stats["rtt_compliance"]
            if current_count > 0:
                stats["rtt_compliance"] = ((current_compliance * (current_count - 1)) + within_18) / current_count
            else:
                stats["rtt_compliance"] = within_18

    def _schedule_bilateral_surgery(self, patient: Patient) -> Optional[Patient]:
        if not patient.is_bilateral or not patient.inter_eye_interval:
            return None
        second_referral = patient.actual_surgery_date + pd.Timedelta(days=patient.inter_eye_interval)
        second_eye = self._create_patient(second_referral)
        second_eye.patient_id = f"{patient.patient_id}_R2"
        second_eye.is_bilateral = False
        second_eye.bilateral_pair = patient.patient_id
        second_eye.is_second_eye = True
        second_eye.surgery_provider = patient.surgery_provider
        second_eye.assessment_provider = patient.assessment_provider
        second_eye.provider_type = patient.provider_type
        second_eye.is_independent_sector = patient.is_independent_sector
        return second_eye

    # --- run & summary ---
    def run_simulation(self, duration_days=90, arrival_method="autocorrelated", warmup_days=180, backlog_days=60):
        print("Starting cataract simulation...")
        print(f"Duration: {duration_days} days | Start: {self.start_date.date()}")
        print(f"Arrivals: {arrival_method} | Backlog: {backlog_days} | Warm-up: {warmup_days}")

        self.setup_arrival_schedule(duration_days, backlog_days, arrival_method, warmup_days)
        self.env.process(self.patient_arrival_process(backlog_days, warmup_days, duration_days))
        self.env.process(self.surgery_scheduling_process(warmup_days))

        total_days = warmup_days + duration_days
        self.env.run(until=total_days)
        print("Simulation complete.")
        return self.get_simulation_summary()

    def get_simulation_summary(self) -> Dict:
        all_patients = self.patients
        measured_start = self.measurement_start or self.start_date
        completed = [p for p in all_patients if p.completed and p.actual_surgery_date and p.actual_surgery_date >= measured_start]
        pending = [p for p in all_patients if not p.completed and not p.cancelled]

        total_arrivals = sum(d.get("arrivals", 0) for d in self.daily_stats)
        total_cap_plan = sum(d["capacity"] for d in self.daily_stats)
        total_cap_eff = sum(d["effective_capacity"] for d in self.daily_stats)
        total_completed = sum(d["patients_completed"] for d in self.daily_stats)

        waits = [p.actual_wait_days for p in completed if p.actual_wait_days is not None]
        if waits:
            rtt_compliance = sum(1 for w in waits if w <= 126) / len(waits) * 100
            mean_wait = float(np.mean(waits))
            median_wait = float(np.median(waits))
        else:
            rtt_compliance = 0.0
            mean_wait = 0.0
            median_wait = 0.0

        fifo_violations, total_pairs = 0, 0
        buckets = defaultdict(list)
        for p in completed:
            buckets[(p.priority, p.served_bucket or "unknown")].append(p)
        for _, group in buckets.items():
            done_sorted = sorted(group, key=lambda x: x.actual_surgery_date)
            for i in range(len(done_sorted)):
                for j in range(i + 1, len(done_sorted)):
                    p1, p2 = done_sorted[i], done_sorted[j]
                    total_pairs += 1
                    if p1.referral_date > p2.referral_date:
                        fifo_violations += 1
        fifo_fairness = (1 - fifo_violations / total_pairs) * 100 if total_pairs > 0 else 100.0

        priority_dist: Dict[int, int] = {}
        for p in all_patients:
            priority_dist[p.priority] = priority_dist.get(p.priority, 0) + 1

        first_eyes = [p for p in all_patients if not p.is_second_eye]
        bilateral_rate = (sum(1 for p in first_eyes if p.is_bilateral) / len(first_eyes) * 100.0) if first_eyes else 0.0

        if self.daily_stats:
            sim_end = self.measurement_start + pd.Timedelta(days=len(self.daily_stats))
        else:
            sim_end = self.start_date

        pending_days_waiting = []
        for p in pending:
            dw = (sim_end - p.referral_date).days
            if dw > 0:
                pending_days_waiting.append(dw)

        complexity_distribution: Dict[int, int] = {}
        hrg_distribution: Dict[str, int] = {}
        for p in all_patients:
            complexity_distribution[p.complexity_level] = complexity_distribution.get(p.complexity_level, 0) + 1
            hrg_distribution[p.hrg_code] = hrg_distribution.get(p.hrg_code, 0) + 1

        provider_loyalty_rate = (sum(1 for p in all_patients if p.provider_loyalty) / len(all_patients) * 100.0) if all_patients else 0.0
        independent_sector_rate = (sum(1 for p in all_patients if p.is_independent_sector) / len(all_patients) * 100.0) if all_patients else 0.0

        assessment_provider_dist: Dict[str, int] = {}
        for p in all_patients:
            short = p.assessment_provider.split()[0] if p.assessment_provider else "Unknown"
            assessment_provider_dist[short] = assessment_provider_dist.get(short, 0) + 1

        surgery_provider_dist: Dict[str, int] = {}
        for p in all_patients:
            short = p.surgery_provider.split()[0] if p.surgery_provider else "Unknown"
            surgery_provider_dist[short] = surgery_provider_dist.get(short, 0) + 1

        provider_performance: Dict[str, Dict] = {}
        for p in completed:
            perf = provider_performance.setdefault(p.surgery_provider, {
                "patients": 0, "total_wait": 0, "within_18_weeks": 0, "avg_wait": 0, "compliance": 0
            })
            perf["patients"] += 1
            perf["total_wait"] += p.actual_wait_days or 0
            if p.within_18_weeks():
                perf["within_18_weeks"] += 1

        for provider, perf in provider_performance.items():
            if perf["patients"] > 0:
                perf["avg_wait"] = perf["total_wait"] / perf["patients"]
                perf["compliance"] = (perf["within_18_weeks"] / perf["patients"]) * 100.0

        return {
            "simulation_period": len(self.daily_stats),
            "total_patients": len(all_patients),
            "completed_surgeries": len(completed),
            "pending_surgeries": len(pending),
            "total_arrivals": total_arrivals,
            "avg_daily_arrivals": total_arrivals / len(self.daily_stats) if self.daily_stats else 0.0,
            "total_capacity_planned": total_cap_plan,
            "total_capacity_effective": total_cap_eff,
            "avg_daily_capacity_effective": (total_cap_eff / len([d for d in self.daily_stats if d["effective_capacity"] > 0])) if any(d["effective_capacity"] > 0 for d in self.daily_stats) else 0.0,
            "capacity_utilization": (total_completed / total_cap_eff * 100.0) if total_cap_eff > 0 else 0.0,
            "mean_current_wait": mean_wait,
            "median_current_wait": median_wait,
            "rtt_compliance_pct": rtt_compliance,
            "fifo_fairness_pct": fifo_fairness,
            "pending_days_waiting": float(np.mean(pending_days_waiting)) if pending_days_waiting else 0.0,
            "priority_distribution": priority_dist,
            "bilateral_rate_pct": bilateral_rate,
            "total_queue_overflow": sum(d.get("queue_overflow", 0) for d in self.daily_stats),
            "avg_daily_overflow": float(np.mean([d.get("queue_overflow", 0) for d in self.daily_stats])) if self.daily_stats else 0.0,
            "complexity_distribution": complexity_distribution,
            "hrg_distribution": hrg_distribution,
            "provider_loyalty_rate_pct": provider_loyalty_rate,
            "independent_sector_rate_pct": independent_sector_rate,
            "top_assessment_providers": dict(sorted(assessment_provider_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_surgery_providers": dict(sorted(surgery_provider_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
            "provider_performance": provider_performance,
        }

    def print_simulation_summary(self):
        s = self.get_simulation_summary()
        print("\n" + "=" * 80)
        print("CATARACT SIMPY SIMULATION SUMMARY")
        print("=" * 80)
        print(f"Simulation Period: {s['simulation_period']} days")
        print(f"Total Patients: {s['total_patients']:,}")
        print(f"Completed Surgeries (measured): {s['completed_surgeries']:,}")
        print(f"Pending Surgeries: {s['pending_surgeries']:,}")
        print(f"Average Daily Arrivals: {s['avg_daily_arrivals']:.1f}")
        print(f"Average Daily Effective Capacity: {s['avg_daily_capacity_effective']:.1f}")
        print(f"Capacity Utilization: {s['capacity_utilization']:.1f}%")
        print("\nWAIT TIME")
        print(f"Mean: {s['mean_current_wait']:.1f} days | Median: {s['median_current_wait']:.1f} days")
        print(f"18-week Compliance: {s['rtt_compliance_pct']:.1f}%")
        print("\nFIFO FAIRNESS")
        print(f"FIFO Adherence: {s['fifo_fairness_pct']:.1f}%")
        print(f"Queue Overflow: {s['avg_daily_overflow']:.1f} patients/day")
        print("\nPROVIDERS (top 5)")
        for i, (provider, count) in enumerate(list(s["top_surgery_providers"].items())[:5], 1):
            pct = count / s["total_patients"] * 100.0
            print(f"  {i}. {provider}: {count:,} ({pct:.1f}%)")
        print("=" * 80)

    def export_results(self, filename_prefix="simpy_cataract_simulation_with_providers"):
        patients_data = []
        for p in self.patients:
            patients_data.append({
                "patient_id": p.patient_id,
                "referral_date": p.referral_date,
                "surgery_date": p.surgery_target_date,
                "actual_surgery_date": p.actual_surgery_date,
                "wait_days": (p.surgery_target_date - p.referral_date).days,
                "actual_wait_days": p.actual_wait_days,
                "rtt_target_wait": (p.surgery_target_date - p.referral_date).days,
                "within_18_weeks": p.within_18_weeks(),
                "hrg_code": p.hrg_code,
                "hrg_description": p.hrg_description,
                "priority": p.priority,
                "priority_name": p.priority_name,
                "complexity_level": p.complexity_level,
                "complexity_category": p.complexity_category,
                "base_surgery_duration_minutes": p.base_surgery_duration_minutes,
                "assessment_provider": p.assessment_provider,
                "surgery_provider": p.surgery_provider,
                "is_independent_sector": p.is_independent_sector,
                "provider_loyalty": p.provider_loyalty,
                "provider_type": p.provider_type,
                "estimated_daily_capacity": p.estimated_daily_capacity,
                "is_bilateral": p.is_bilateral,
                "inter_eye_interval": p.inter_eye_interval,
                "bilateral_pair": p.bilateral_pair,
                "is_second_eye": p.is_second_eye,
                "surgery_duration": p.surgery_duration,
                "preop_assessment": p.preop_assessment,
                "postop_1week": p.postop_1week,
                "postop_6week": p.postop_6week,
                "theatre_setup": p.theatre_setup,
                "theatre_changeover": p.theatre_changeover,
                "total_theatre_time": p.total_theatre_time,
                "total_episode_time": p.total_episode_time,
                "scheduled": True,
                "completed": p.completed,
                "cancelled": p.cancelled,
                "rescheduled": False,
                "served_bucket": p.served_bucket,
            })
        patients_df = pd.DataFrame(patients_data)
        daily_df = pd.DataFrame(self.daily_stats)

        provider_stats_data = []
        for provider, stats in self.provider_stats.items():
            provider_stats_data.append({
                "provider_name": provider,
                "total_assessments": stats["total_assessments"],
                "total_surgeries": stats["total_surgeries"],
                "total_patients": stats["total_patients"],
                "avg_wait_days": stats["avg_wait_days"],
                "rtt_compliance": stats["rtt_compliance"],
                "is_independent": stats["is_independent"],
                "estimated_capacity": stats["estimated_capacity"],
            })
        provider_stats_df = pd.DataFrame(provider_stats_data)

        pfile = f"{filename_prefix}_patients.csv"
        dfile = f"{filename_prefix}_daily_stats.csv"
        prfile = f"{filename_prefix}_provider_stats.csv"
        patients_df.to_csv(pfile, index=False)
        daily_df.to_csv(dfile, index=False)
        provider_stats_df.to_csv(prfile, index=False)
        print(f"Exported: {pfile}, {dfile}, {prfile}")
        return patients_df, daily_df, provider_stats_df

    def get_provider_summary(self) -> Dict:
        return {
            "total_providers": len(self.provider_stats),
            "provider_loyalty_rate": (sum(1 for p in self.patients if p.provider_loyalty) / len(self.patients) * 100.0) if self.patients else 0.0,
            "independent_sector_rate": (sum(1 for p in self.patients if p.is_independent_sector) / len(self.patients) * 100.0) if self.patients else 0.0,
            "top_4_providers_share": 0.0,
            "provider_stats": dict(self.provider_stats),
        }


# Centralized policies 

class CentralizedTriagePolicy(TriagePolicy):
    def __init__(
        self,
        capacity_weight: float = 0.4,
        workload_weight: float = 0.3,
        complexity_weight: float = 0.2,
        distance_weight: float = 0.1,
        max_assignments_per_day: Optional[int] = None,
    ):
        self.capacity_weight = capacity_weight
        self.workload_weight = workload_weight
        self.complexity_weight = complexity_weight
        self.distance_weight = distance_weight
        self.max_assignments_per_day = max_assignments_per_day

        self.daily_reassignments = defaultdict(int)
        self.provider_workload = defaultdict(int)

        self.provider_capabilities = {
            "SPAMEDICA": {"max_complexity": 9, "specialty_score": 0.9},
            "NEWMEDICA": {"max_complexity": 8, "specialty_score": 0.85},
            "FITZWILLIAM": {"max_complexity": 8, "specialty_score": 0.85},
            "CAMBRIDGE": {"max_complexity": 7, "specialty_score": 0.8},
            "ANGLIA": {"max_complexity": 7, "specialty_score": 0.8},
            "NORTH": {"max_complexity": 7, "specialty_score": 0.8},
            "WEST": {"max_complexity": 6, "specialty_score": 0.75},
            "COMMUNITY": {"max_complexity": 5, "specialty_score": 0.7},
        }

    def assign_provider(self, patient, date: pd.Timestamp, sim) -> Optional[str]:
        date_key = date.date()
        if date_key not in self.daily_reassignments:
            self.daily_reassignments.clear()
            self.daily_reassignments[date_key] = 0
        if self.max_assignments_per_day and self.daily_reassignments[date_key] >= self.max_assignments_per_day:
            return None

        available_providers = list(sim._known_providers)
        if not available_providers:
            return None

        self._update_workload_tracking(sim)

        scores = {prov: self._score(patient, prov, sim) for prov in available_providers}
        if not scores:
            return None

        best_provider = max(scores, key=lambda p: scores[p])
        current_score = scores.get(patient.surgery_provider, 0)
        best_score = scores[best_provider]

        if best_provider != patient.surgery_provider and best_score > current_score * 1.1:
            self.daily_reassignments[date_key] += 1
            self.provider_workload[best_provider] += 1
            if patient.surgery_provider in self.provider_workload:
                self.provider_workload[patient.surgery_provider] -= 1
            return best_provider
        return None

    def _update_workload_tracking(self, sim):
        self.provider_workload.clear()
        for p in sim.waiting_queue:
            if not p.completed and not p.cancelled:
                self.provider_workload[p.surgery_provider] += 1

    def _score(self, patient, provider: str, sim) -> float:
        score = 0.0
        capacity_est = sim._provider_cap_estimate.get(provider, 1.0)
        score += self.capacity_weight * min(1.0, capacity_est / 50.0)

        max_wl = max(self.provider_workload.values()) if self.provider_workload else 1
        wl = self.provider_workload.get(provider, 0)
        wl_score = 1.0 - (wl / max_wl) if max_wl > 0 else 1.0
        score += self.workload_weight * wl_score

        score += self.complexity_weight * self._complexity_match(patient, provider)
        score += self.distance_weight * self._distance_score(provider)
        return score

    def _complexity_match(self, patient, provider: str) -> float:
        c = patient.complexity_level
        cap = self.provider_capabilities.get(provider, {"max_complexity": 5, "specialty_score": 0.7})
        max_c = cap["max_complexity"]
        spec = cap["specialty_score"]

        if c <= max_c:
            bonus = min(1.0, c / 9.0)
            return spec * (0.8 + 0.2 * bonus)
        penalty = (c - max_c) / 9.0
        return max(0.1, spec - penalty)

    def _distance_score(self, provider: str) -> float:
        accessibility = {
            "CAMBRIDGE": 0.9,
            "ANGLIA": 0.85,
            "NORTH": 0.8,
            "SPAMEDICA": 0.75,
            "NEWMEDICA": 0.75,
            "FITZWILLIAM": 0.7,
            "WEST": 0.8,
            "COMMUNITY": 0.95,
        }
        return accessibility.get(provider, 0.7)

    def get_policy_stats(self) -> Dict:
        total = sum(self.daily_reassignments.values())
        days = len(self.daily_reassignments)
        return {
            "policy_name": "CentralizedTriage",
            "total_reassignments": total,
            "avg_daily_reassignments": total / max(1, days),
            "active_days": days,
            "current_workload_distribution": dict(self.provider_workload),
            "provider_capabilities": self.provider_capabilities,
        }


class CapacityBalancedTriagePolicy(TriagePolicy):
    def __init__(self, rebalance_threshold: float = 0.2):
        self.rebalance_threshold = rebalance_threshold
        self.daily_assignments = defaultdict(int)

    def assign_provider(self, patient, date: pd.Timestamp, sim) -> Optional[str]:
        providers = list(sim._known_providers)
        if len(providers) <= 1:
            return None

        best_provider = None
        best_capacity = -1.0
        current_capacity = sim._provider_cap_estimate.get(patient.surgery_provider, 1.0)

        for prov in providers:
            cap = sim._provider_cap_estimate.get(prov, 1.0)
            if cap > best_capacity:
                best_capacity = cap
                best_provider = prov

        if best_provider and best_provider != patient.surgery_provider and best_capacity > current_capacity * (1 + self.rebalance_threshold):
            return best_provider
        return None


def compare_triage_policies(baseline_results: Dict, centralized_results: Dict) -> Dict:
    return {
        "wait_time_improvement": {
            "mean_wait_change": centralized_results["mean_current_wait"] - baseline_results["mean_current_wait"],
            "median_wait_change": centralized_results["median_current_wait"] - baseline_results["median_current_wait"],
            "compliance_change": centralized_results["rtt_compliance_pct"] - baseline_results["rtt_compliance_pct"],
        },
        "capacity_utilization": {
            "baseline": baseline_results["capacity_utilization"],
            "centralized": centralized_results["capacity_utilization"],
            "improvement": centralized_results["capacity_utilization"] - baseline_results["capacity_utilization"],
        },
        "fairness_metrics": {
            "baseline_fifo": baseline_results["fifo_fairness_pct"],
            "centralized_fifo": centralized_results["fifo_fairness_pct"],
            "overflow_reduction": baseline_results["avg_daily_overflow"] - centralized_results["avg_daily_overflow"],
        },
        "provider_distribution": {
            "baseline_top4_share": sum(list(baseline_results["top_surgery_providers"].values())[:4]) / baseline_results["total_patients"] if baseline_results["total_patients"] else 0.0,
            "centralized_top4_share": sum(list(centralized_results["top_surgery_providers"].values())[:4]) / centralized_results["total_patients"] if centralized_results["total_patients"] else 0.0,
        },
    }


def run_triage_comparison():
    print("TRIAGE POLICY COMPARISON")
    print("=" * 80)

    print("\n1) Baseline (NoopPolicy)")
    baseline_sim = CataractSimulationSimPy(start_date="2024-03-01", random_seed=42)
    baseline_sim.set_policy(NoopPolicy())
    baseline_results = baseline_sim.run_simulation(
        duration_days=365,
        arrival_method="autocorrelated",
        backlog_days=60,
        warmup_days=180,
    )

    print("\n2) Centralized triage")
    centralized_sim = CataractSimulationSimPy(start_date="2024-03-01", random_seed=42)
    centralized_policy = CentralizedTriagePolicy(
        capacity_weight=0.4,
        workload_weight=0.3,
        complexity_weight=0.2,
        distance_weight=0.1,
    )
    centralized_sim.set_policy(centralized_policy)
    centralized_results = centralized_sim.run_simulation(
        duration_days=365,
        arrival_method="autocorrelated",
        backlog_days=60,
        warmup_days=180,
    )

    print("\n3) Comparison")
    print("=" * 60)
    comparison = compare_triage_policies(baseline_results, centralized_results)
    print("WAIT TIMES")
    print(f"  Mean change: {comparison['wait_time_improvement']['mean_wait_change']:+.1f} days")
    print(f"  Median change: {comparison['wait_time_improvement']['median_wait_change']:+.1f} days")
    print(f"  RTT compliance change: {comparison['wait_time_improvement']['compliance_change']:+.1f}%")
    print("\nCAPACITY UTILIZATION")
    print(f"  Baseline: {comparison['capacity_utilization']['baseline']:.1f}%")
    print(f"  Centralized: {comparison['capacity_utilization']['centralized']:.1f}%")
    print(f"  Improvement: {comparison['capacity_utilization']['improvement']:+.1f}%")
    print("\nFAIRNESS & DISTRIBUTION")
    print(f"  Overflow reduction: {comparison['fairness_metrics']['overflow_reduction']:+.1f} patients/day")
    print(f"  Top 4 share - Baseline: {comparison['provider_distribution']['baseline_top4_share']:.1f}%")
    print(f"  Top 4 share - Centralized: {comparison['provider_distribution']['centralized_top4_share']:.1f}%")

    policy_stats = centralized_policy.get_policy_stats()
    print("\nCENTRALIZED TRIAGE ACTIVITY")
    print(f"  Total reassignments: {policy_stats['total_reassignments']:,}")
    print(f"  Avg daily reassignments: {policy_stats['avg_daily_reassignments']:.1f}")

    return {
        "baseline_results": baseline_results,
        "centralized_results": centralized_results,
        "comparison": comparison,
        "policy_stats": policy_stats,
    }


if __name__ == "__main__":

    sim = CataractSimulationSimPy(start_date="2024-03-01", random_seed=42)
    sim.set_policy(AntiBreachISPolicy(days_to_breach_threshold=21))
    summary = sim.run_simulation(duration_days=365, arrival_method="autocorrelated", backlog_days=60, warmup_days=180)
    sim.print_simulation_summary()
    sim.export_results()
    provider_summary = sim.get_provider_summary()
    print("\nPROVIDER SUMMARY")
    print(f"Total providers: {provider_summary['total_providers']}")
    print(f"Provider loyalty rate: {provider_summary['provider_loyalty_rate']:.1f}%")
    print(f"Independent sector rate: {provider_summary['independent_sector_rate']:.1f}%")
    results = run_triage_comparison()
