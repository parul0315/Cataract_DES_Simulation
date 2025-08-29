import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class CataractProviderGenerator:
    """
    Provider generator for DES cataract model 
    """
    
    def __init__(self, matched_data: pd.DataFrame = None):
        """
        Initialize provider generator with data observed
        """
        if matched_data is not None:
            self.matched_data = matched_data
            self._analyze_from_data()
        else:
            self._load_analyzed_parameters()
        
    def _load_analyzed_parameters(self):
        """Load the pre-analyzed parameters from data analysis"""
        
        # Assessment provider probabilitie 
        self.assessment_providers = [
            'NORTH WEST ANGLIA NHS FOUNDATION TRUST',
            'ANGLIA COMMUNITY EYE SERVICE LTD', 
            'SPAMEDICA PETERBOROUGH',
            'CAMBRIDGE UNIVERSITY HOSPITALS NHS FOUNDATION TRUST',
            'SPAMEDICA BEDFORD',
            'FITZWILLIAM HOSPITAL',
            'COMMUNITY HEALTH AND EYECARE LIMITED',
            'NEWMEDICA SUFFOLK - BURY ST EDMUNDS - CLARITY HOUSE',
            'SPAMEDICA CHELMSFORD',
            'WEST SUFFOLK NHS FOUNDATION TRUST'
        ]
        
        self.assessment_probs = np.array([
            0.2309, 0.2292, 0.2233, 0.2152, 0.0583,
            0.0172, 0.0045, 0.0032, 0.0030, 0.0026
        ])
        # Normalize remaining probability for other providers
        remaining_prob = 1.0 - self.assessment_probs.sum()
        other_providers_count = 20  # 30 total - 10 listed
        if other_providers_count > 0:
            other_prob = remaining_prob / other_providers_count
            other_providers = [f"OTHER_PROVIDER_{i+1}" for i in range(other_providers_count)]
            self.assessment_providers.extend(other_providers)
            other_probs = np.full(other_providers_count, other_prob)
            self.assessment_probs = np.concatenate([self.assessment_probs, other_probs])
        
        # Surgery provider probabilities 
        self.surgery_providers = self.assessment_providers.copy()
        self.surgery_probs = self.assessment_probs.copy()
        
        # Provider loyalty rate 
        self.provider_loyalty_rate = 0.95  # 95% continue with same provider
        
        # Independent sector probability
        self.independent_sector_prob = 0.5502
        
        # Provider type mapping 
        self.independent_providers = {
            'ANGLIA COMMUNITY EYE SERVICE LTD': True,
            'SPAMEDICA PETERBOROUGH': True,
            'SPAMEDICA BEDFORD': True,
            'FITZWILLIAM HOSPITAL': True,
            'COMMUNITY HEALTH AND EYECARE LIMITED': True,
            'NEWMEDICA SUFFOLK - BURY ST EDMUNDS - CLARITY HOUSE': True,
            'SPAMEDICA CHELMSFORD': True,
            'NORTH WEST ANGLIA NHS FOUNDATION TRUST': False,
            'CAMBRIDGE UNIVERSITY HOSPITALS NHS FOUNDATION TRUST': False,
            'WEST SUFFOLK NHS FOUNDATION TRUST': False
        }
    
    def _analyze_from_data(self):
        """Analyze provider patterns from historical data (if provided)."""
        
        # Assessment provider probabilities
        assess_counts = self.matched_data['Provider_x'].value_counts()
        self.assessment_providers = list(assess_counts.index)
        self.assessment_probs = (assess_counts / len(self.matched_data)).values
        
        # Surgery provider probabilities
        surg_counts = self.matched_data['Provider_y'].value_counts()
        self.surgery_providers = list(surg_counts.index)
        self.surgery_probs = (surg_counts / len(self.matched_data)).values
        
        # Calculate provider loyalty rate
        same_provider = (self.matched_data['Provider_x'] == self.matched_data['Provider_y']).sum()
        self.provider_loyalty_rate = same_provider / len(self.matched_data)
        
        # Independent sector probability
        if 'Independent Sector_y' in self.matched_data.columns:
            self.independent_sector_prob = (self.matched_data['Independent Sector_y'] == 'Yes').mean()
        else:
            self.independent_sector_prob = 0.55
            
        # Provider type mapping
        if 'Independent Sector_y' in self.matched_data.columns:
            self.independent_providers = {}
            for provider in self.surgery_providers:
                provider_data = self.matched_data[self.matched_data['Provider_y'] == provider]
                is_independent = (provider_data['Independent Sector_y'] == 'Yes').mean() > 0.5
                self.independent_providers[provider] = is_independent
    
    def generate_assessment_provider(self):
        """Generate a random assessment provider based on historical probabilities."""
        return np.random.choice(self.assessment_providers, p=self.assessment_probs)
    
    def generate_surgery_provider(self, assessment_provider: Optional[str] = None):
        """
        Generate surgery provider based on assessment provider (with high loyalty rate).
        """
        if assessment_provider and np.random.random() < self.provider_loyalty_rate: 
            if assessment_provider in self.surgery_providers:
                return assessment_provider
        
        # Patient switches providers or assessment provider doesn't do surgery
        return np.random.choice(self.surgery_providers, p=self.surgery_probs)
    
    def is_independent_sector(self, provider: str) -> bool:
        """Determine if a provider is independent sector."""
        if provider in self.independent_providers:
            return self.independent_providers[provider]
        
        # For unknown providers, use overall probability
        return np.random.random() < self.independent_sector_prob
    
    def generate_complete_pathway(self) -> Dict[str, any]:
        """
        Generate a complete patient pathway through the system.
        """
        # Generate assessment provider
        assessment_provider = self.generate_assessment_provider()
        
        # Generate surgery provider (with loyalty consideration)
        surgery_provider = self.generate_surgery_provider(assessment_provider)
        
        # Determine sector type
        is_independent = self.is_independent_sector(surgery_provider)
        
        # Estimate capacity (simplified model)
        base_capacity = self._estimate_provider_capacity(surgery_provider)
        
        return {
            'assessment_provider': assessment_provider,
            'surgery_provider': surgery_provider,
            'is_independent_sector': is_independent,
            'provider_loyalty': assessment_provider == surgery_provider,
            'estimated_daily_capacity': base_capacity,
            'provider_type': 'Independent' if is_independent else 'NHS'
        }
    
    def _estimate_provider_capacity(self, provider: str) -> float:
        """
        Estimate daily surgical capacity based on provider size.
        Based on your data showing daily averages.
        """
        # Major providers 
        major_providers = [
            'NORTH WEST ANGLIA NHS FOUNDATION TRUST',
            'ANGLIA COMMUNITY EYE SERVICE LTD', 
            'SPAMEDICA PETERBOROUGH',
            'CAMBRIDGE UNIVERSITY HOSPITALS NHS FOUNDATION TRUST'
        ]
        
        if provider in major_providers:
            return np.random.normal(10, 2) 
        
        # Medium providers
        elif 'SPAMEDICA' in provider or 'NEWMEDICA' in provider:
            return np.random.normal(4, 1)   
        
        # Smaller providers
        else:
            return np.random.normal(2, 0.5)  
    
    def get_system_stats(self) -> Dict[str, any]:
        """statistics about the provider"""
        return {
            'total_assessment_providers': len(self.assessment_providers),
            'total_surgery_providers': len(self.surgery_providers),
            'provider_loyalty_rate': self.provider_loyalty_rate,
            'independent_sector_rate': self.independent_sector_prob,
            'top_assessment_provider': self.assessment_providers[0],
            'top_surgery_provider': self.surgery_providers[0],
            'market_concentration': sum(self.assessment_probs[:4]) 
        }
    
    def simulate_patient_batch(self, n_patients: int) -> pd.DataFrame:
        """
        Simulate provider assignments for a batch of patients
        """
        results = []
        
        for i in range(n_patients):
            pathway = self.generate_complete_pathway()
            pathway['patient_id'] = f"SIM_{i+1:06d}"
            results.append(pathway)
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Initialize generator 
    provider_gen = CataractProviderGenerator()
    stats = provider_gen.get_system_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    for i in range(5):
        pathway = provider_gen.generate_complete_pathway()
        print(f"Patient {i+1}:")
        print(f"  Assessment: {pathway['assessment_provider'][:40]}")
        print(f"  Surgery: {pathway['surgery_provider'][:40]}")
        print(f"  Same Provider: {pathway['provider_loyalty']}")
        print(f"  Sector: {pathway['provider_type']}")
        print(f"  Daily Capacity: {pathway['estimated_daily_capacity']:.1f}")
        print()
    
    # Validate with larger simulation
    sim_results = provider_gen.simulate_patient_batch(1000)
    
    print(f"Provider loyalty rate: {sim_results['provider_loyalty'].mean():.1%} (expected: 95%)")
    print(f"Independent sector rate: {sim_results['is_independent_sector'].mean():.1%} (expected: 55%)")
    
    print(f"Top 3 assessment providers:")
    assess_counts = sim_results['assessment_provider'].value_counts()
    for provider, count in assess_counts.head(3).items():
        expected = provider_gen.assessment_probs[provider_gen.assessment_providers.index(provider)]
        print(f"  {provider[:40]}...: {count/1000:.1%} (expected: {expected:.1%})")
    
    print(f"Top 3 surgery providers:")
    surg_counts = sim_results['surgery_provider'].value_counts()  
    for provider, count in surg_counts.head(3).items():
        if provider in provider_gen.surgery_providers:
            expected = provider_gen.surgery_probs[provider_gen.surgery_providers.index(provider)]
            print(f"  {provider[:40]}...: {count/1000:.1%} (expected: {expected:.1%})")