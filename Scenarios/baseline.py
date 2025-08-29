import pandas as pd
import numpy as np
import simpy
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

from capacity_generator import CataractCapacityGenerator
from patient_generator import CataractPatientGenerator
from rtt_generator import RTTWaitTimeGenerator
from service_time_generator import CataractServiceTimeGenerator
from arrival_generator import HealthcareArrivalGenerator
from provider import CataractProviderGenerator

@dataclass
class Patient:
    """Patient data structure"""
    patient_id: str
    referral_date: pd.Timestamp
    surgery_target_date: pd.Timestamp  # RTT
    priority: int
    priority_name: str
    hrg_code: str
    hrg_description: str  
    complexity_level: int 
    complexity_category: str
    
    # Surgery times
    surgery_duration: float
    preop_assessment: float
    postop_1week: float
    postop_6week: float
    theatre_setup: float
    theatre_changeover: float
    total_theatre_time: float
    total_episode_time: float
    base_surgery_duration_minutes: int 
    
    # Provider information
    assessment_provider: str
    surgery_provider: str
    is_independent_sector: bool
    provider_loyalty: bool  
    provider_type: str  
    estimated_daily_capacity: float
    
    # Bilateral info
    is_bilateral: bool = False
    inter_eye_interval: Optional[int] = None
    bilateral_pair: Optional[str] = None
    is_second_eye: bool = False
    
    # Simulation state
    actual_surgery_date: Optional[pd.Timestamp] = None
    actual_wait_days: Optional[int] = None
    served_bucket: Optional[str] = None
    completed: bool = False
    cancelled: bool = False
    
    def days_waiting(self, current_date: pd.Timestamp) -> int:
        """Calculate days waiting from referral to current date"""
        return (current_date - self.referral_date).days
    
    def days_until_target(self, current_date: pd.Timestamp) -> int:
        """Calculate days until RTT target is breached"""
        return (self.surgery_target_date - current_date).days
    
    def is_overdue(self, current_date: pd.Timestamp) -> bool:
        """Check if patient is past their RTT target"""
        return current_date >= self.surgery_target_date
    
    def within_18_weeks(self) -> bool:
        """Check if completed within 18 weeks (126 days)"""
        if self.actual_wait_days is not None:
            return self.actual_wait_days <= 126
        return (self.surgery_target_date - self.referral_date).days <= 126


class CataractSimulationSimPy:
    """
    SimPy-based cataract surgery simulation with RTT-SLA scheduling, complexity integration AND provider pathways
    """
    
    def __init__(self, start_date='2024-01-01', random_seed=42):
        # Set random seed
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Initialize SimPy environment
        self.env = simpy.Environment()
        
        # Initialize generators 
        self.init_generators()
        
        # Simulation parameters
        self.start_date = pd.Timestamp(start_date)
        self.min_lead_time_days = 14
        self.early_pull_window_days = 28
        self.early_pull_fraction = 0.15
        
        # Capacity parameters
        self.capacity_disruption_mean = 0.92
        self.capacity_disruption_sd = 0.025
        
        # Scheduling parameters
        self.frac_at_risk = 0.25
        self.frac_due = 0.35
        self.wait_threshold_days = 126  # 18 weeks
        self.compliance_target = 0.69
        
        # State tracking
        self.patients = []
        self.completed_patients = []
        self.daily_stats = []
        self.patient_counter = 0
        self.daily_capacity = {}
        self.daily_arrivals_schedule = {}
        self.measurement_start = None
        
        # Provider tracking
        self.provider_stats = defaultdict(lambda: {
            'total_assessments': 0,
            'total_surgeries': 0,
            'total_patients': 0,
            'avg_wait_days': 0,
            'rtt_compliance': 0,
            'is_independent': False,
            'estimated_capacity': 0
        })
        
        self.waiting_queue = deque()
        
        # Setup capacity
        self.setup_yearly_capacity(2024)
        
    def init_generators(self):
        """Initializing parameter generators"""
        self.capacity_gen = CataractCapacityGenerator()
        self.patient_gen = CataractPatientGenerator(random_seed=self.random_seed)
        self.rtt_gen = RTTWaitTimeGenerator()
        self.service_gen = CataractServiceTimeGenerator()
        self.arrival_gen = HealthcareArrivalGenerator()
        self.arrival_gen.set_random_seed(self.random_seed)
        self.provider_gen = CataractProviderGenerator()
        
    
    def setup_yearly_capacity(self, year: int):
        """Setup daily capacity for the year"""
        print(f"Setting up capacity for {year}...")
        total_capacity = 0
        capacity_days = 0
        
        for month in range(1, 13):
            monthly_schedule = self.capacity_gen.generate_monthly_schedule(month, year)
            for _, row in monthly_schedule.iterrows():
                date_key = row['date'].date()
                base_capacity = row['capacity']
                # Apply calibration factor
                realistic_capacity = max(0, int(base_capacity * 0.80))
                self.daily_capacity[date_key] = realistic_capacity
                
                if realistic_capacity > 0:
                    capacity_days += 1
                    total_capacity += realistic_capacity
        
        avg_capacity = total_capacity / capacity_days if capacity_days > 0 else 0
        print(f"Capacity setup: {capacity_days} operating days, avg {avg_capacity:.1f} surgeries/day")
    
    def setup_arrival_schedule(self, duration_days: int, backlog_days: int, method: str, warmup_days: int):
        """Setup arrival schedule matching original implementation"""
        total_days = backlog_days + warmup_days + duration_days
        schedule_start = self.start_date - pd.Timedelta(days=(backlog_days + warmup_days))

        print(f"Generating arrivals using '{method}' method")
        arrival_ts = self.arrival_gen.generate_arrivals_time_series(
            start_date=schedule_start.strftime('%Y-%m-%d'),
            num_days=total_days,
            method=method
        )

        self.daily_arrivals_schedule = {}
        if len(arrival_ts) > 0:
            daily_counts = arrival_ts.groupby('date').size().to_dict()
            current = schedule_start
            for _ in range(total_days):
                self.daily_arrivals_schedule[current.date()] = int(daily_counts.get(current.date(), 0))
                current += pd.Timedelta(days=1)
        else:
            current = schedule_start
            for _ in range(total_days):
                self.daily_arrivals_schedule[current.date()] = int(self.arrival_gen.get_expected_daily_arrivals(current.weekday()))
                current += pd.Timedelta(days=1)

        total_arrivals = sum(self.daily_arrivals_schedule.values())
        print(f"Arrivals: {total_arrivals:,} over {total_days} days (avg {total_arrivals/total_days:.1f}/day)")
    
    def create_patient(self, referral_date: pd.Timestamp) -> Patient:
        """Create a new patient with provider information"""
        self.patient_counter += 1
        patient_id = f"P{self.patient_counter:06d}"
        
        # Generate patient data
        pdata = self.patient_gen.generate_patient()
        
        # Generate provider pathway
        provider_pathway = self.provider_gen.generate_complete_pathway()
        
        # Bilateral surgery
        is_bilateral = np.random.random() < 0.398
        inter_eye_interval = max(30, int(np.random.lognormal(3.7, 1.2))) if is_bilateral else None
        
        # RTT target
        rtt_wait = self.rtt_gen.generate_rtt_wait(
            priority=pdata['priority'],
            hrg=pdata['hrg_code'],
            referral_month=referral_date.month
        )
        surgery_target_date = referral_date + pd.Timedelta(days=rtt_wait)
        
        # Service times
        etimes = self.service_gen.generate_complete_episode_times(
            hrg_code=pdata['hrg_code'],
            complexity=pdata['complexity_category'].lower(),
            activity_type='Day Case'
        )
        
        # Create patient instance
        patient = Patient(
            patient_id=patient_id,
            referral_date=referral_date,
            surgery_target_date=surgery_target_date,
            priority=pdata['priority'],
            priority_name=pdata['priority_name'],
            hrg_code=pdata['hrg_code'],
            hrg_description=pdata['hrg_description'],
            complexity_level=pdata['complexity_level'],
            complexity_category=pdata['complexity_category'],
            surgery_duration=etimes['surgery_duration'],
            preop_assessment=etimes['preop_assessment'],
            postop_1week=etimes['postop_1week'],
            postop_6week=etimes['postop_6week'],
            theatre_setup=etimes['theatre_setup'],
            theatre_changeover=etimes['theatre_changeover'],
            total_theatre_time=etimes['total_theatre_time'],
            total_episode_time=etimes['total_episode_time'],
            base_surgery_duration_minutes=pdata['base_surgery_duration_minutes'],
            assessment_provider=provider_pathway['assessment_provider'],
            surgery_provider=provider_pathway['surgery_provider'],
            is_independent_sector=provider_pathway['is_independent_sector'],
            provider_loyalty=provider_pathway['provider_loyalty'],
            provider_type=provider_pathway['provider_type'],
            estimated_daily_capacity=provider_pathway['estimated_daily_capacity'],
            is_bilateral=is_bilateral,
            inter_eye_interval=inter_eye_interval
        )
        
        self.update_provider_stats(patient)
        
        return patient
    
    def update_provider_stats(self, patient: Patient):
        """Tracking provider statistics"""
        # Assessment provider stats
        self.provider_stats[patient.assessment_provider]['total_assessments'] += 1
        self.provider_stats[patient.assessment_provider]['is_independent'] = patient.is_independent_sector
        
        # Surgery provider stats 
        self.provider_stats[patient.surgery_provider]['total_surgeries'] += 1
        self.provider_stats[patient.surgery_provider]['total_patients'] += 1
        self.provider_stats[patient.surgery_provider]['is_independent'] = patient.is_independent_sector
        self.provider_stats[patient.surgery_provider]['estimated_capacity'] = patient.estimated_daily_capacity
    
    def patient_arrival_process(self, backlog_days: int, warmup_days: int, duration_days: int):
        """SimPy process for patient arrivals - generates backlog then continues with scheduled arrivals"""
        # Generate backlog patients
        backlog_start = -(backlog_days + warmup_days)
        backlog_count = 0
        
        for day_offset in range(backlog_start, -warmup_days):
            arrival_date = self.start_date + pd.Timedelta(days=day_offset)
            daily_arrivals = self.daily_arrivals_schedule.get(arrival_date.date(), 
                                                            self.arrival_gen.get_expected_daily_arrivals(arrival_date.weekday()))
            
            for _ in range(daily_arrivals):
                patient = self.create_patient(arrival_date)
                self.patients.append(patient)
                self.waiting_queue.append(patient)
                backlog_count += 1
        
        print(f"Created {backlog_count} backlog patients")
        
        # Continue arrivals during simulation period
        for day in range(warmup_days + duration_days):
            arrival_date = self.start_date + pd.Timedelta(days=day - warmup_days)
            daily_arrivals = self.daily_arrivals_schedule.get(arrival_date.date(),
                                                            self.arrival_gen.get_expected_daily_arrivals(arrival_date.weekday()))
            
            for _ in range(daily_arrivals):
                patient = self.create_patient(arrival_date)
                self.patients.append(patient)
                self.waiting_queue.append(patient)
                
                # Log first few patients with provider info
                if len(self.patients) <= 5:
                    print(f"Patient {patient.patient_id}: Priority={patient.priority}, "
                          f"HRG={patient.hrg_code}, Complexity={patient.complexity_level} ({patient.complexity_category}), "
                          f"Assessment={patient.assessment_provider[:20]}..., "
                          f"Surgery={patient.surgery_provider[:20]}..., "
                          f"Provider loyalty={patient.provider_loyalty}, "
                          f"Sector={patient.provider_type}")
            
            # Wait until next day
            yield self.env.timeout(1)
    
    def surgery_scheduling_process(self, warmup_days: int):
        """SimPy process for daily surgery scheduling"""
        measurement_start_day = warmup_days
        self.measurement_start = self.start_date  # Setting measurement start
        
        while True:
            current_day = self.env.now
            current_date = self.start_date + pd.Timedelta(days=current_day - warmup_days)
            
            # today's capacity
            base_capacity = self.daily_capacity.get(current_date.date(), 0)
            
            if base_capacity > 0:
                stats = self.schedule_daily_surgeries(current_date, base_capacity)
                stats['arrivals'] = self.daily_arrivals_schedule.get(current_date.date(), 0)
                
                # Recording statistics if in measurement period
                if current_day >= measurement_start_day:
                    self.daily_stats.append(stats)
                
                # Log progress with provider info 
                if current_day >= measurement_start_day and (current_day - measurement_start_day) % 30 == 0:
                    day_measured = current_day - measurement_start_day
                    # Show provider distribution of ready patients
                    ready_patients = self._get_eligible_patients(current_date)
                    provider_dist = {}
                    for p in ready_patients:
                        short_name = p.surgery_provider.split()[0] if p.surgery_provider else 'UNK'
                        provider_dist[short_name] = provider_dist.get(short_name, 0) + 1
                    
                    provider_str = ', '.join([f"{k}:{v}" for k, v in sorted(provider_dist.items())])
                    print(f"Day {day_measured}: {current_date.strftime('%Y-%m-%d')}, "
                          f"Patients ready for surgery: {stats['patients_ready']}, "
                          f"Providers: [{provider_str}], "
                          f"Capacity: {stats['effective_capacity']}, "
                          f"Completed: {stats['patients_completed']}")
            
            yield self.env.timeout(1)  
    
    def schedule_daily_surgeries(self, date: pd.Timestamp, base_capacity: int) -> Dict:
        """Schedule surgeries for a given day"""
        # Apply capacity disruption
        effective_capacity = self._effective_capacity(base_capacity)
        
        if effective_capacity == 0:
            return {
                'date': date, 'capacity': 0, 'effective_capacity': 0,
                'patients_ready': 0, 'patients_completed': 0, 'patients_cancelled': 0,
                'total_surgery_time': 0, 'total_theatre_time': 0,
                'utilization': 0.0, 'avg_surgery_time': 0.0, 'queue_overflow': 0,
                'complexity_distribution': {}, 'provider_distribution': {}
            }
        
        # eligible patients for surgery
        eligible_patients = self._get_eligible_patients(date)
        
        # Categorize patients into buckets
        due, at_risk, near_due, fifo_early = self._categorize_patients(eligible_patients, date)
        
        # Schedule patients according to priority buckets
        scheduled_patients = self._select_patients_for_surgery(
            due, at_risk, near_due, fifo_early, effective_capacity
        )
        
        # Perform surgeries, track complexity and providers
        completed_count = 0
        total_surgery_time = 0
        total_theatre_time = 0
        complexity_completed = {}
        provider_completed = {}
        
        for patient in scheduled_patients:
            self._complete_surgery(patient, date)
            completed_count += 1
            total_surgery_time += patient.surgery_duration
            total_theatre_time += patient.total_theatre_time
            
            # Track complexity distribution
            complexity_completed[patient.complexity_level] = complexity_completed.get(patient.complexity_level, 0) + 1
            
            # Track provider distribution
            provider_short = patient.surgery_provider.split()[0] if patient.surgery_provider else 'UNK'
            provider_completed[provider_short] = provider_completed.get(provider_short, 0) + 1
            
            # Handle bilateral cases
            if patient.is_bilateral and patient.inter_eye_interval:
                second_eye = self._schedule_bilateral_surgery(patient)
                if second_eye:
                    self.patients.append(second_eye)
                    self.waiting_queue.append(second_eye)
        
        utilization = min(100.0, (total_theatre_time / (effective_capacity * 60.0)) * 100.0)
        avg_surgery_time = total_surgery_time / completed_count if completed_count > 0 else 0.0
        
        return {
            'date': date,
            'capacity': base_capacity,
            'effective_capacity': effective_capacity,
            'patients_ready': len(due),
            'patients_completed': completed_count,
            'patients_cancelled': 0,
            'total_surgery_time': total_surgery_time,
            'total_theatre_time': total_theatre_time,
            'utilization': utilization,
            'avg_surgery_time': avg_surgery_time,
            'queue_overflow': max(0, len(due) - effective_capacity),
            'complexity_distribution': complexity_completed,
            'provider_distribution': provider_completed
        }
    
    def _effective_capacity(self, base_capacity: int) -> int:
        """Apply capacity disruption factor"""
        if base_capacity <= 0:
            return 0
        
        factor = np.clip(
            np.random.normal(self.capacity_disruption_mean, self.capacity_disruption_sd),
            0.75, 1.00
        )
        return max(1, int(np.floor(base_capacity * factor)))
    
    def _get_eligible_patients(self, date: pd.Timestamp) -> List[Patient]:
        """Get patients eligible for surgery"""
        min_lead = pd.Timedelta(days=self.min_lead_time_days)
        eligible = []
        
        for patient in list(self.waiting_queue):
            if (not patient.completed and not patient.cancelled and 
                date >= patient.referral_date + min_lead):
                eligible.append(patient)
        
        return eligible
    
    def _categorize_patients(self, eligible: List[Patient], date: pd.Timestamp):
        """Categorize patients into scheduling buckets"""
        due = []
        at_risk = []
        near_due = []
        fifo_early = []
        taken = set()
        
        # 1. Due/overdue patients
        for patient in eligible:
            if patient.is_overdue(date):
                due.append(patient)
                taken.add(patient.patient_id)
        
        # 2. At-risk patients 
        for patient in eligible:
            if patient.patient_id in taken:
                continue
            waited = patient.days_waiting(date)
            if 115 <= waited < 126:
                at_risk.append(patient)
                taken.add(patient.patient_id)
        
        # 3. Near-due patients
        for patient in eligible:
            if patient.patient_id in taken:
                continue
            days_to_target = patient.days_until_target(date)
            if 0 <= days_to_target <= self.early_pull_window_days:
                near_due.append(patient)
                taken.add(patient.patient_id)
        
        # 4. Everyone else 
        for patient in eligible:
            if patient.patient_id not in taken:
                fifo_early.append(patient)
        
        # Sort within buckets 
        def fifo_key(p):
            return (-p.priority, p.referral_date, p.complexity_level) 
        
        due.sort(key=fifo_key)
        at_risk.sort(key=fifo_key)
        near_due.sort(key=fifo_key)
        fifo_early.sort(key=fifo_key)
        
        return due, at_risk, near_due, fifo_early
    
    def _select_patients_for_surgery(self, due: List, at_risk: List, near_due: List, 
                                   fifo_early: List, effective_capacity: int):
        """Select patients for surgery based on capacity allocation"""
        scheduled = []
        
        # 1. Due/overdue first
        cap_due = min(int(round(self.frac_due * effective_capacity)), effective_capacity)
        take = min(len(due), cap_due)
        for i in range(take):
            due[i].served_bucket = 'due'
            scheduled.append(due[i])
        
        # 2. At-risk second
        remaining = effective_capacity - len(scheduled)
        if remaining > 0:
            cap_risk = min(int(round(self.frac_at_risk * effective_capacity)), remaining)
            take = min(len(at_risk), cap_risk)
            for i in range(take):
                at_risk[i].served_bucket = 'at_risk'
                scheduled.append(at_risk[i])
        
        # 3. Near-due (limited early pull)
        remaining = effective_capacity - len(scheduled)
        if remaining > 0 and near_due:
            quota = min(int(self.early_pull_fraction * effective_capacity), remaining, len(near_due))
            for i in range(quota):
                near_due[i].served_bucket = 'near_due'
                scheduled.append(near_due[i])
        
        # 4. Fill remaining with FIFO
        remaining = effective_capacity - len(scheduled)
        if remaining > 0 and fifo_early:
            take = min(remaining, len(fifo_early))
            for i in range(take):
                fifo_early[i].served_bucket = 'fifo'
                scheduled.append(fifo_early[i])
        
        return scheduled
    
    def _complete_surgery(self, patient: Patient, date: pd.Timestamp):
        """Complete surgery for a patient"""
        patient.completed = True
        patient.actual_surgery_date = date
        patient.actual_wait_days = patient.days_waiting(date)
        
        # Remove from waiting queue
        if patient in self.waiting_queue:
            self.waiting_queue.remove(patient)
        
        self.completed_patients.append(patient)
        
        # Update provider stats with completion data
        if patient.surgery_provider in self.provider_stats:
            stats = self.provider_stats[patient.surgery_provider]
            # Update running average of wait times
            current_avg = stats['avg_wait_days']
            current_count = stats['total_patients']
            if current_count > 0:
                stats['avg_wait_days'] = ((current_avg * (current_count - 1)) + patient.actual_wait_days) / current_count
            else:
                stats['avg_wait_days'] = patient.actual_wait_days
            
            # Update RTT compliance
            within_18_weeks = patient.within_18_weeks()
            current_compliance = stats['rtt_compliance']
            if current_count > 0:
                stats['rtt_compliance'] = ((current_compliance * (current_count - 1)) + (1 if within_18_weeks else 0)) / current_count
            else:
                stats['rtt_compliance'] = 1 if within_18_weeks else 0
    
    def _schedule_bilateral_surgery(self, patient: Patient):
        """Schedule second eye surgery for bilateral patient"""
        if not patient.is_bilateral or not patient.inter_eye_interval:
            return None
        
        second_referral = patient.actual_surgery_date + pd.Timedelta(days=patient.inter_eye_interval)
        second_eye = self.create_patient(second_referral)
        second_eye.patient_id = f"{patient.patient_id}_R2"
        second_eye.is_bilateral = False
        second_eye.bilateral_pair = patient.patient_id
        second_eye.is_second_eye = True
        
        return second_eye
    
    def run_simulation(self, duration_days=90, arrival_method='autocorrelated', warmup_days=180, backlog_days=60):
        """Run the SimPy simulation with full validation setup"""
        print("NHS SimPy Cataract Surgery Simulation")
        print(f"Duration: {duration_days} days, Start: {self.start_date.strftime('%Y-%m-%d')}")
        print(f"Method: {arrival_method}, Backlog: {backlog_days} days, Warm-up: {warmup_days} days")
        
        # Setup arrival schedule
        self.setup_arrival_schedule(duration_days, backlog_days, arrival_method, warmup_days)
        
        # Start the processes
        self.env.process(self.patient_arrival_process(backlog_days, warmup_days, duration_days))
        self.env.process(self.surgery_scheduling_process(warmup_days))
        
        # Run simulation
        total_days = warmup_days + duration_days
        self.env.run(until=total_days)
        
        print("SimPy simulation completed.")
        
        return self.get_simulation_summary()
    
    def get_simulation_summary(self) -> Dict:
        """Generate comprehensive simulation summary"""
        all_patients = self.patients
        measured_start = self.measurement_start or self.start_date

        # Filter to measurement period patients
        completed = [p for p in all_patients if p.completed and p.actual_surgery_date is not None and p.actual_surgery_date >= measured_start]
        pending = [p for p in all_patients if not p.completed and not p.cancelled]

        # Basic metrics
        total_arrivals = sum(d.get('arrivals', 0) for d in self.daily_stats)
        total_cap_plan = sum(d['capacity'] for d in self.daily_stats)
        total_cap_eff = sum(d['effective_capacity'] for d in self.daily_stats)
        total_completed = sum(d['patients_completed'] for d in self.daily_stats)

        # Wait time analysis
        waits = [p.actual_wait_days for p in completed if p.actual_wait_days is not None]
        if waits:
            rtt_compliance = sum(1 for w in waits if w <= 126) / len(waits) * 100
            mean_wait = float(np.mean(waits))
            median_wait = float(np.median(waits))
        else:
            rtt_compliance = 0.0
            mean_wait = 0.0
            median_wait = 0.0

        # FIFO Fairness Analysis
        fifo_violations, total_pairs = 0, 0
        buckets = defaultdict(list)
        for p in completed:
            buckets[(p.priority, p.served_bucket or 'unknown')].append(p)
        
        for (_, _), group in buckets.items():
            group_sorted_by_done = sorted(group, key=lambda x: x.actual_surgery_date)
            for i in range(len(group_sorted_by_done)):
                for j in range(i+1, len(group_sorted_by_done)):
                    p1, p2 = group_sorted_by_done[i], group_sorted_by_done[j]
                    total_pairs += 1
                    if p1.referral_date > p2.referral_date:
                        fifo_violations += 1
        
        fifo_fairness = (1 - fifo_violations/total_pairs) * 100 if total_pairs > 0 else 100.0

        # Priority distribution
        priority_dist = {}
        for p in all_patients:
            priority_dist[p.priority] = priority_dist.get(p.priority, 0) + 1

        # Bilateral rate
        first_eyes = [p for p in all_patients if not p.is_second_eye]
        bilateral_rate = (sum(1 for p in first_eyes if p.is_bilateral) / len(first_eyes) * 100.0) if first_eyes else 0.0

        # Pending patient wait times
        if self.daily_stats:
            sim_end = self.measurement_start + pd.Timedelta(days=len(self.daily_stats))
        else:
            sim_end = self.start_date
        
        pending_days_waiting = []
        for p in pending:
            days_waiting = (sim_end - p.referral_date).days
            if days_waiting > 0:  # Only include positive wait times
                pending_days_waiting.append(days_waiting)

        # Complexity analysis
        complexity_distribution = {}
        hrg_distribution = {}
        for p in all_patients:
            complexity_distribution[p.complexity_level] = complexity_distribution.get(p.complexity_level, 0) + 1
            hrg_distribution[p.hrg_code] = hrg_distribution.get(p.hrg_code, 0) + 1

        # Provider pathway analysis
        provider_loyalty_count = sum(1 for p in all_patients if p.provider_loyalty)
        provider_loyalty_rate = (provider_loyalty_count / len(all_patients) * 100.0) if all_patients else 0.0
        
        independent_sector_count = sum(1 for p in all_patients if p.is_independent_sector)
        independent_sector_rate = (independent_sector_count / len(all_patients) * 100.0) if all_patients else 0.0
        
        # Top assessment providers
        assessment_provider_dist = {}
        for p in all_patients:
            short_name = p.assessment_provider.split()[0] if p.assessment_provider else 'Unknown'
            assessment_provider_dist[short_name] = assessment_provider_dist.get(short_name, 0) + 1
        
        # Top surgery providers
        surgery_provider_dist = {}
        for p in all_patients:
            short_name = p.surgery_provider.split()[0] if p.surgery_provider else 'Unknown'
            surgery_provider_dist[short_name] = surgery_provider_dist.get(short_name, 0) + 1
        
        # Provider performance analysis (completed patients only)
        provider_performance = {}
        for p in completed:
            if p.surgery_provider not in provider_performance:
                provider_performance[p.surgery_provider] = {
                    'patients': 0,
                    'total_wait': 0,
                    'within_18_weeks': 0,
                    'avg_wait': 0,
                    'compliance': 0
                }
            
            perf = provider_performance[p.surgery_provider]
            perf['patients'] += 1
            perf['total_wait'] += p.actual_wait_days or 0
            if p.within_18_weeks():
                perf['within_18_weeks'] += 1
        
        # Calculate averages
        for provider, perf in provider_performance.items():
            if perf['patients'] > 0:
                perf['avg_wait'] = perf['total_wait'] / perf['patients']
                perf['compliance'] = (perf['within_18_weeks'] / perf['patients']) * 100.0

        return {
            # Original metrics
            'simulation_period': len(self.daily_stats),
            'total_patients': len(all_patients),
            'completed_surgeries': len(completed),
            'pending_surgeries': len(pending),
            'total_arrivals': total_arrivals,
            'avg_daily_arrivals': total_arrivals / len(self.daily_stats) if self.daily_stats else 0.0,
            'total_capacity_planned': total_cap_plan,
            'total_capacity_effective': total_cap_eff,
            'avg_daily_capacity_effective': (total_cap_eff / len([d for d in self.daily_stats if d['effective_capacity'] > 0])) if any(d['effective_capacity'] > 0 for d in self.daily_stats) else 0.0,
            'capacity_utilization': (total_completed / total_cap_eff * 100.0) if total_cap_eff > 0 else 0.0,
            'mean_current_wait': mean_wait,
            'median_current_wait': median_wait,
            'rtt_compliance_pct': rtt_compliance,
            'fifo_fairness_pct': fifo_fairness,
            'pending_days_waiting': float(np.mean(pending_days_waiting)) if pending_days_waiting else 0.0,
            'priority_distribution': priority_dist,
            'bilateral_rate_pct': bilateral_rate,
            'total_queue_overflow': sum(d.get('queue_overflow', 0) for d in self.daily_stats),
            'avg_daily_overflow': float(np.mean([d.get('queue_overflow', 0) for d in self.daily_stats])) if self.daily_stats else 0.0,
            'complexity_distribution': complexity_distribution,
            'hrg_distribution': hrg_distribution,
            # Provider metrics
            'provider_loyalty_rate_pct': provider_loyalty_rate,
            'independent_sector_rate_pct': independent_sector_rate,
            'top_assessment_providers': dict(sorted(assessment_provider_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_surgery_providers': dict(sorted(surgery_provider_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
            'provider_performance': provider_performance
        }
    
    def print_simulation_summary(self):
        """Print formatted simulation summary with provider section"""
        s = self.get_simulation_summary()
        
        print("Baseline cataract simulation summary")
        print(f"Simulation Period: {s['simulation_period']} days")
        print(f"Total Patients: {s['total_patients']:,}")
        print(f"Completed Surgeries (measured): {s['completed_surgeries']:,}")
        print(f"Pending Surgeries: {s['pending_surgeries']:,}")
        print(f"Average Daily Arrivals: {s['avg_daily_arrivals']:.1f}")
        print(f"Average Daily Effective Capacity: {s['avg_daily_capacity_effective']:.1f}")
        print(f"Capacity Utilization: {s['capacity_utilization']:.1f}%")

        print(f"\nWAIT TIME ANALYSIS (measured window):")
        print(f"Mean Actual Wait: {s['mean_current_wait']:.1f} days")
        print(f"Median Actual Wait: {s['median_current_wait']:.1f} days")
        print(f"18-week Compliance: {s['rtt_compliance_pct']:.1f}%")

        print(f"\nFIFO FAIRNESS (within bucket & priority):")
        print(f"FIFO Adherence: {s['fifo_fairness_pct']:.1f}%")
        print(f"Queue Overflow (due/overdue): {s['avg_daily_overflow']:.1f} patients/day")

        # Complexity section
        print(f"\nCOMPLEXITY DISTRIBUTION:")
        for level in sorted(s['complexity_distribution'].keys()):
            count = s['complexity_distribution'][level]
            pct = count / s['total_patients'] * 100.0
            print(f"  Level {level}: {count:,} ({pct:.1f}%)")

        print(f"\nHRG DISTRIBUTION (Top 5):")
        hrg_sorted = sorted(s['hrg_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]
        for hrg, count in hrg_sorted:
            pct = count / s['total_patients'] * 100.0
            print(f"  {hrg}: {count:,} ({pct:.1f}%)")

        # Provider analysis section
        print(f"Patient continuity rate with same Provider: {s['provider_loyalty_rate_pct']:.1f}% (target: ~95%)")
        print(f"Independent Sector Rate: {s['independent_sector_rate_pct']:.1f}% (target: ~55%)")
        
        print(f"\nTop assessment providers:")
        for i, (provider, count) in enumerate(list(s['top_assessment_providers'].items())[:5], 1):
            pct = count / s['total_patients'] * 100.0
            print(f"  {i}. {provider}: {count:,} ({pct:.1f}%)")
        
        print(f"\nTop surgery providers:")
        for i, (provider, count) in enumerate(list(s['top_surgery_providers'].items())[:5], 1):
            pct = count / s['total_patients'] * 100.0
            print(f"  {i}. {provider}: {count:,} ({pct:.1f}%)")
        
        print(f"\nProvider performance (Top 5 by volume - completed patients only):")
        perf_sorted = sorted([(k, v) for k, v in s['provider_performance'].items() if v['patients'] > 10], 
                           key=lambda x: x[1]['patients'], reverse=True)[:5]
        for provider, perf in perf_sorted:
            short_name = provider.split()[0] if len(provider.split()) > 0 else provider[:20]
            print(f"  {short_name}: {perf['patients']} patients, "
                  f"{perf['avg_wait']:.1f}d avg wait, "
                  f"{perf['compliance']:.1f}% RTT compliance")

        print(f"\nOther metrics:")
        print(f"Bilateral Rate (first eyes): {s['bilateral_rate_pct']:.1f}%")
        print(f"Pending Average Wait: {s['pending_days_waiting']:.1f} days")
        
        # Priority distribution
        priority_names = {1: 'Routine', 2: 'Urgent', 3: 'Emergency'}
        for priority, count in s['priority_distribution'].items():
            pct = count / s['total_patients'] * 100.0
            name = priority_names.get(priority, f'Priority {priority}')
            print(f"  {name}: {count:,} ({pct:.1f}%)")

        # Validation status
        print("\nValidation Status:")
        if 90 <= s['mean_current_wait'] <= 120:
            print(f"  Wait Times: {s['mean_current_wait']:.1f} days ")
        else:
            print(f" Wait Times: {s['mean_current_wait']:.1f} days ")

        if 65 <= s['rtt_compliance_pct'] <= 75:
            print(f" RTT Compliance: {s['rtt_compliance_pct']:.1f}%")
        else:
            print(f" RTT Compliance: {s['rtt_compliance_pct']:.1f}%")

        if 37 <= s['bilateral_rate_pct'] <= 42:
            print(f" Bilateral Rate: {s['bilateral_rate_pct']:.1f}% ")
        else:
            print(f" Bilateral Rate: {s['bilateral_rate_pct']:.1f}% ")

        if s['fifo_fairness_pct'] >= 90:
            print(f" FIFO Fairness: {s['fifo_fairness_pct']:.1f}% ")
        else:
            print(f" FIFO Fairness: {s['fifo_fairness_pct']:.1f}%")

        # Provider validation
        if 90 <= s['provider_loyalty_rate_pct'] <= 98:
            print(f" Provider Loyalty: {s['provider_loyalty_rate_pct']:.1f}% ")
        else:
            print(f" Provider Loyalty: {s['provider_loyalty_rate_pct']:.1f}% ")

        if 50 <= s['independent_sector_rate_pct'] <= 60:
            print(f" Independent Sector: {s['independent_sector_rate_pct']:.1f}%")
        else:
            print(f" Independent Sector: {s['independent_sector_rate_pct']:.1f}%")

        print("="*80)

    def export_results(self, filename_prefix='simpy_cataract_simulation_with_providers'):
        """Export simulation results to CSV files with provider fields"""
        # Convert patients to DataFrame format with provider columns
        patients_data = []
        for p in self.patients:
            patients_data.append({
                'patient_id': p.patient_id,
                'referral_date': p.referral_date,
                'surgery_date': p.surgery_target_date,
                'actual_surgery_date': p.actual_surgery_date,
                'wait_days': (p.surgery_target_date - p.referral_date).days,
                'actual_wait_days': p.actual_wait_days,
                'rtt_target_wait': (p.surgery_target_date - p.referral_date).days,
                'within_18_weeks': p.within_18_weeks(),
                'hrg_code': p.hrg_code,
                'hrg_description': p.hrg_description,
                'priority': p.priority,
                'priority_name': p.priority_name,
                'complexity_level': p.complexity_level,
                'complexity_category': p.complexity_category,
                'base_surgery_duration_minutes': p.base_surgery_duration_minutes,
                # Provider fields
                'assessment_provider': p.assessment_provider,
                'surgery_provider': p.surgery_provider,
                'is_independent_sector': p.is_independent_sector,
                'provider_loyalty': p.provider_loyalty,
                'provider_type': p.provider_type,
                'estimated_daily_capacity': p.estimated_daily_capacity,
                'is_bilateral': p.is_bilateral,
                'inter_eye_interval': p.inter_eye_interval,
                'bilateral_pair': p.bilateral_pair,
                'is_second_eye': p.is_second_eye,
                'surgery_duration': p.surgery_duration,
                'preop_assessment': p.preop_assessment,
                'postop_1week': p.postop_1week,
                'postop_6week': p.postop_6week,
                'theatre_setup': p.theatre_setup,
                'theatre_changeover': p.theatre_changeover,
                'total_theatre_time': p.total_theatre_time,
                'total_episode_time': p.total_episode_time,
                'scheduled': True,
                'completed': p.completed,
                'cancelled': p.cancelled,
                'rescheduled': False,
                'served_bucket': p.served_bucket
            })
        
        patients_df = pd.DataFrame(patients_data)
        daily_df = pd.DataFrame(self.daily_stats)
        
        # Export provider stats
        provider_stats_data = []
        for provider, stats in self.provider_stats.items():
            provider_stats_data.append({
                'provider_name': provider,
                'total_assessments': stats['total_assessments'],
                'total_surgeries': stats['total_surgeries'],
                'total_patients': stats['total_patients'],
                'avg_wait_days': stats['avg_wait_days'],
                'rtt_compliance': stats['rtt_compliance'],
                'is_independent': stats['is_independent'],
                'estimated_capacity': stats['estimated_capacity']
            })
        
        provider_stats_df = pd.DataFrame(provider_stats_data)
        
        pfile = f"{filename_prefix}_patients.csv"
        dfile = f"{filename_prefix}_daily_stats.csv"
        prfile = f"{filename_prefix}_provider_stats.csv"
        
        patients_df.to_csv(pfile, index=False)
        daily_df.to_csv(dfile, index=False)
        provider_stats_df.to_csv(prfile, index=False)
        
        print(f"Results exported to {pfile}, {dfile}, and {prfile}")
        return patients_df, daily_df, provider_stats_df

    def get_provider_summary(self) -> Dict:
        """detailed provider pathway summary"""
        return {
            'total_providers': len(self.provider_stats),
            'provider_loyalty_rate': (sum(1 for p in self.patients if p.provider_loyalty) / len(self.patients) * 100.0) if self.patients else 0.0,
            'independent_sector_rate': (sum(1 for p in self.patients if p.is_independent_sector) / len(self.patients) * 100.0) if self.patients else 0.0,
            'top_4_providers_share': 0.0,  # Could calculate market concentration
            'provider_stats': dict(self.provider_stats)
        }


if __name__ == "__main__":
    sim = CataractSimulationSimPy(start_date='2024-03-01', random_seed=42)
    
    summary = sim.run_simulation(
        duration_days=365,
        arrival_method='autocorrelated',
        backlog_days=60,
        warmup_days=180
    )
    
    sim.print_simulation_summary()
    
    # Export results 
    patients_df, daily_df, provider_stats_df = sim.export_results()
    
    provider_summary = sim.get_provider_summary()
    print(f"\nProvider validation:")
    print(f"Total providers in system: {provider_summary['total_providers']}")
    print(f"Provider loyalty rate: {provider_summary['provider_loyalty_rate']:.1f}%")
    print(f"Independent sector rate: {provider_summary['independent_sector_rate']:.1f}%")
