# Cataract_DES_Simulation
## Simulation-Based Evaluation of a Single PTL Model for Cataract Surgery

This repository contains the full implementation of a discrete-event simulation (DES) used to study cataract surgery waiting list management within the Cambridgeshire and Peterborough Integrated Care System (ICS). The project evaluates whether consolidating multiple provider-specific Patient Treatment Lists (PTLs) into a single regional waiting list can improve waiting times, capacity utilisation, and equity across providers.

The work is based on the MSc dissertation project by Parul Nagar (2025).

## Overview

Cataract surgery is one of the highest-volume elective procedures in the NHS, yet waiting times remain uneven across regions and providers. Provider-specific PTLs create bottlenecks even when system-wide capacity exists elsewhere.

This simulation model recreates the full cataract pathway—from referral to discharge—using real operational data and explores what happens when the PTL is unified at system level. Alongside the baseline model, the repository includes scenarios for centralised triage and early independent-sector integration.

The core of the project is a SimPy-based DES that models thousands of patients, each with their own HRG code, complexity, RTT target, provider behaviour, and bilateral surgery probability.

## Repository Structure
cataract-des-simulation
│
├── Parameter Generators/
│   ├── Arrival_generator.py          # Daily arrival patterns, weekday effects, seasonality, autocorrelation
│   ├── capacity_generator.py         # Theatre slot generation with weekday + seasonal behaviour
│   ├── patient_generator.py          # Case-mix, HRGs, priority classes, bilateral behaviour
│   ├── provider.py                   # Provider selection, loyalty patterns, NHS vs Independent split
│   ├── rtt_generator.py              # Referral-to-treatment wait distribution generator
│   └── service_time_generator.py     # Surgery duration, assessments, overhead and admin timings
│
├── Scenarios/
│   ├── baseline.py                   # Baseline system simulation (current real-world pathway)
│   ├── centralisedtriage.py          # Scenario with standardised referral + centralised triage
│   └── des_simulation.ipynb          # Notebook for running any scenario with interactive exploration
│
├── patientlevelanalysis.ipynb        # Patient-level EDA on ICS datasets
├── README.md                         # This file
└── requirements.txt                  # Dependencies

## Core Concepts
### The Cataract Pathway Modelled

The simulation follows the standard three-step NHS cataract pathway:

1. Outpatient Assessment

2. Cataract Surgery (Day Case)

3. Post-Operative Follow-Up

Patients may require second-eye surgery, which re-enters the pathway realistically after a generated inter-eye interval.

### Key Features of the Simulation

- End-to-end DES built with SimPy

- Stochastic arrivals with weekday patterns, seasonal multipliers and lag-1 autocorrelation

- HRG-based complexity influencing duration, risk and follow-up load

- RTT-aware scheduling prioritising near-breach and overdue patients

- Provider behaviour modelling, including loyalty (staying with assessment provider)

- Independent sector integration, with real market-share proportions

- Bilateral surgery logic, with empirically calibrated return intervals

- Daily capacity generation based on ICS data

### Monte Carlo replications for scenario-level comparison

Scenarios Included
1. Baseline

A faithful representation of the current ICS cataract pathway:

- separate PTLs across providers

- variable capacity

- patient loyalty

- real HRG distributions

- RTT patterns embedded

Used for validation against observed ICS metrics.

2. Centralised Triage

Simulates a system-wide triage process:

- uniform referral standards

- consistent prioritisation

- improved alignment of demand and capacity

- more clinically appropriate ordering of cases

3. Early Independent-Sector Integration

- Independent providers are included earlier in the pathway:

- suitable patients diverted immediately post-triage

- reduces NHS backlog growth

- tests system-wide capacity pooling

How to Run the Simulation
Option 1: Notebook Interface

Open:

Scenarios/des_simulation.ipynb


Then:

Select scenario

Run the model

View visualisations like wait distributions, utilisation curves, RTT compliance, and scenario comparisons

Option 2: Command Line

Run individual scenarios:

python Scenarios/baseline.py
python Scenarios/centralisedtriage.py


Each script prints:

mean + percentile wait times

RTT compliance

utilisation metrics

complexity mix

provider distribution

backlog evolution

Outputs & Metrics

The simulation reports:

Waiting-Time Metrics

mean / median

P50 / P75 / P90 / P95 / P99

distribution plots

RTT 18-week compliance

Capacity & Theatre Metrics

daily theatre minutes used

utilisation percentage

case durations by HRG

Provider-Level Dynamics

volume share

loyalty patterns

independent-sector participation

Scenario Comparisons

reduction in backlog size

improvements in compliance

change in theatre efficiency

shifts in case complexity over time

Installation

Install all dependencies:

pip install -r requirements.txt


Main packages:

SimPy

NumPy

Pandas

SciPy

Matplotlib / Seaborn

Jupyter

Data Sources (not included in repo)

This model uses:

Patient-level data from the C&P ICS (referrals, HRG codes, RTT timestamps, provider information)

NOD audit reports for complexity and risk validation

ICS surgical-time evidence (theatre duration by complexity)

National ophthalmology guideline timings

Due to NHS data governance, patient-level data is not shared here.

## Purpose of This Repository

This repository makes the simulation framework fully reproducible, allowing:

NHS analysts to evaluate backlog policies

Researchers to extend the DES to other elective pathways

Students to learn real-world healthcare simulation modelling

ICS teams to explore triage or capacity reallocation strategies before implementation

## Citation

If you use this work, please cite:

Parul Nagar (2025)
Simulation-Based Evaluation of a Single PTL Model for Cataract Surgery in the Cambridgeshire and Peterborough ICS
MSc Dissertation, University of Bristol.
