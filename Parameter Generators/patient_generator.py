import numpy as np
import pandas as pd
from scipy import stats
import random
from typing import Dict, List, Tuple, Optional

class CataractPatientGenerator:
    """
    Patient mix and complexity generator for cataract simulation
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize generator with real data distributions"""
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # HRG Distribution 
        hrg_raw = {
            'BZ33Z': 29.0,   # Minor, Cataract or Lens Procedures
            'BZ34B': 28.7,   # Phacoemulsification with complications  
            'BZ34C': 24.2,   # Phacoemulsification standard
            'BZ34A': 6.7,    # Phacoemulsification complex
            'BZ31A': 6.1,    # Very Major with complications
            'BZ31B': 4.2,    # Very Major standard
            'BZ30A': 0.4,    # Complex with complications
            'BZ30B': 0.4,    # Complex standard
            'BZ32B': 0.3,    # Intermediate with complications
            'BZ32A': 0.1     # Intermediate standard
        }
        # Normalizing to ensure sum = 1.0
        total = sum(hrg_raw.values())
        self.hrg_distribution = {k: v/total for k, v in hrg_raw.items()}
        
        # Priority Distribution 
        priority_raw = {
            1: 97.2,  # Routine
            2: 2.7,   # Urgent  
            3: 0.1    # Emergency (using 0.1 instead of 0.0)
        }
        
        total = sum(priority_raw.values())
        self.priority_distribution = {k: v/total for k, v in priority_raw.items()}
        
        # Bilateral Surgery Parameters
        self.bilateral_rate = 0.398  # 39.8%
        self.inter_eye_mean = 83.6   # days
        self.inter_eye_median = 42.0 # days
        self.inter_eye_std = 111.9   # days
        
        # Fit log-normal distribution for inter-eye intervals
        self.inter_eye_params = self._fit_lognormal_distribution(
            self.inter_eye_median, self.inter_eye_mean, self.inter_eye_std
        )
        
        # HRG Complexity Mapping 
        self.complexity_mapping = {
            'BZ33Z': {'level': 1, 'category': 'Minor', 'base_duration': 15},
            'BZ34C': {'level': 2, 'category': 'Standard_Phaco', 'base_duration': 20},
            'BZ34B': {'level': 3, 'category': 'Phaco_Complications', 'base_duration': 25},
            'BZ34A': {'level': 4, 'category': 'Complex_Phaco', 'base_duration': 30},
            'BZ32A': {'level': 5, 'category': 'Intermediate', 'base_duration': 35},
            'BZ32B': {'level': 5, 'category': 'Intermediate', 'base_duration': 35},
            'BZ31B': {'level': 6, 'category': 'Very_Major', 'base_duration': 40},
            'BZ31A': {'level': 7, 'category': 'Very_Major_Complex', 'base_duration': 50},
            'BZ30B': {'level': 8, 'category': 'Complex', 'base_duration': 60},
            'BZ30A': {'level': 9, 'category': 'Complex_Complications', 'base_duration': 75}
        }
        
        # HRG Descriptions
        self.hrg_descriptions = {
            'BZ33Z': 'Minor, Cataract or Lens Procedures',
            'BZ34B': 'Phacoemulsification Cataract Extraction and Lens I...',
            'BZ34C': 'Phacoemulsification Cataract Extraction and Lens I...',
            'BZ34A': 'Phacoemulsification Cataract Extraction and Lens I...',
            'BZ31A': 'Very Major, Cataract or Lens Procedures, with CC S...',
            'BZ31B': 'Very Major, Cataract or Lens Procedures, with CC S...',
            'BZ30A': 'Complex, Cataract or Lens Procedures, with CC Scor...',
            'BZ30B': 'Complex, Cataract or Lens Procedures, with CC Scor...',
            'BZ32B': 'Intermediate, Cataract or Lens Procedures, with CC...',
            'BZ32A': 'Intermediate, Cataract or Lens Procedures, with CC...'
        }
    
    def _fit_lognormal_distribution(self, median: float, mean: float, std: float) -> Dict[str, float]:
        """
        Fit log-normal distribution parameters from empirical statistics
        Using method of moments for log-normal distribution
        """

        mu = np.log(median)
        
        sigma = np.sqrt(2 * np.log(mean / median))
        
        return {'mu': mu, 'sigma': sigma}
    
    def generate_patient(self) -> Dict:
        """Generate a single patient with all attributes"""
        # Generate HRG code
        hrg_code = self._sample_from_distribution(self.hrg_distribution)
        
        # Generate priority
        priority = self._sample_from_distribution(self.priority_distribution)
        
        # Determine bilateral status
        is_bilateral = np.random.random() < self.bilateral_rate
        
        # Generate inter-eye interval if bilateral
        inter_eye_interval = None
        if is_bilateral:
            # Sample from fitted log-normal distribution
            interval = np.random.lognormal(
                self.inter_eye_params['mu'], 
                self.inter_eye_params['sigma']
            )
            inter_eye_interval = max(1, int(round(interval)))  # At least 1 day, round to nearest day
        
        # Getting complexity info
        complexity_info = self.complexity_mapping[hrg_code]
        
        return {
            'hrg_code': hrg_code,
            'hrg_description': self.hrg_descriptions[hrg_code],
            'complexity_level': complexity_info['level'],
            'complexity_category': complexity_info['category'],
            'priority': priority,
            'priority_name': self._get_priority_name(priority),
            'is_bilateral': is_bilateral,
            'inter_eye_interval_days': inter_eye_interval,
            'base_surgery_duration_minutes': complexity_info['base_duration']
        }
    
    def generate_surgery_duration(self, hrg_code: str, add_variation: bool = True) -> int:
        """
        Generate realistic surgery duration based on HRG complexity
        """
        base_duration = self.complexity_mapping[hrg_code]['base_duration']
        
        if add_variation:
            # Add log-normal variation to base duration
            # Use coefficient of variation of ~0.3 
            cv = 0.3
            sigma = np.sqrt(np.log(1 + cv**2))
            mu = np.log(base_duration) - sigma**2/2
            
            duration = np.random.lognormal(mu, sigma)
            duration = max(5, int(round(duration)))  # Minimum 5 minutes
        else:
            duration = base_duration
            
        return duration
    
    def generate_inter_eye_interval(self) -> int:
        """Generate inter-eye interval for bilateral patients"""
        interval = np.random.lognormal(
            self.inter_eye_params['mu'], 
            self.inter_eye_params['sigma']
        )
        return max(1, int(round(interval)))
    
    def _sample_from_distribution(self, distribution: Dict) -> any:
        """Sample from a probability distribution dictionary"""
        items = list(distribution.keys())
        probabilities = list(distribution.values())
        return np.random.choice(items, p=probabilities)
    
    def _get_priority_name(self, priority: int) -> str:
        """Convert priority number to name"""
        priority_names = {1: 'Routine', 2: 'Urgent', 3: 'Emergency'}
        return priority_names.get(priority, 'Unknown')
    
    def validate_distributions(self, n_samples: int = 10000) -> Dict:
        """
        Validate generator against empirical data
        
        """
        print(f"Validating generator with {n_samples:,} samples...")
        
        # Generate samples
        samples = [self.generate_patient() for _ in range(n_samples)]
        
        # Analyze HRG distribution
        hrg_counts = {}
        priority_counts = {}
        bilateral_count = 0
        intervals = []
        
        for patient in samples:
            # Count HRGs
            hrg = patient['hrg_code']
            hrg_counts[hrg] = hrg_counts.get(hrg, 0) + 1
            
            # Count priorities
            priority = patient['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            # Count bilateral
            if patient['is_bilateral']:
                bilateral_count += 1
                intervals.append(patient['inter_eye_interval_days'])
        
        # Calculate validation metrics
        validation_results = {
            'sample_size': n_samples,
            'hrg_validation': {},
            'priority_validation': {},
            'bilateral_validation': {},
            'interval_validation': {}
        }
        
        # HRG validation
        print("\nHRG Distribution Validation:")
        print(f"{'HRG':<8} {'Expected':<10} {'Actual':<10} {'Diff':<8}")
        print("-" * 40)
        for hrg, expected_prob in self.hrg_distribution.items():
            actual_prob = hrg_counts.get(hrg, 0) / n_samples
            diff = abs(expected_prob - actual_prob)
            validation_results['hrg_validation'][hrg] = {
                'expected': expected_prob,
                'actual': actual_prob,
                'difference': diff
            }
            print(f"{hrg:<8} {expected_prob:<10.3f} {actual_prob:<10.3f} {diff:<8.3f}")
        
        # Priority validation
        print(f"\nPriority Distribution Validation:")
        print(f"{'Priority':<10} {'Expected':<10} {'Actual':<10} {'Diff':<8}")
        print("-" * 42)
        for priority, expected_prob in self.priority_distribution.items():
            actual_prob = priority_counts.get(priority, 0) / n_samples
            diff = abs(expected_prob - actual_prob)
            validation_results['priority_validation'][priority] = {
                'expected': expected_prob,
                'actual': actual_prob,
                'difference': diff
            }
            print(f"{priority:<10} {expected_prob:<10.3f} {actual_prob:<10.3f} {diff:<8.3f}")
        
        # Bilateral validation
        actual_bilateral_rate = bilateral_count / n_samples
        bilateral_diff = abs(self.bilateral_rate - actual_bilateral_rate)
        
        validation_results['bilateral_validation'] = {
            'expected_rate': self.bilateral_rate,
            'actual_rate': actual_bilateral_rate,
            'difference': bilateral_diff,
            'bilateral_count': bilateral_count
        }
        
        print(f"\nBilateral Surgery Validation:")
        print(f"Expected rate: {self.bilateral_rate:.3f}")
        print(f"Actual rate: {actual_bilateral_rate:.3f}")
        print(f"Difference: {bilateral_diff:.3f}")
        
        # Interval validation
        if intervals:
            intervals_array = np.array(intervals)
            validation_results['interval_validation'] = {
                'expected_mean': self.inter_eye_mean,
                'actual_mean': intervals_array.mean(),
                'expected_median': self.inter_eye_median,
                'actual_median': np.median(intervals_array),
                'expected_std': self.inter_eye_std,
                'actual_std': intervals_array.std(),
                'sample_count': len(intervals)
            }
            
            print(f"\nInter-eye Interval Validation ({len(intervals)} bilateral patients):")
            print(f"Mean - Expected: {self.inter_eye_mean:.1f}, Actual: {intervals_array.mean():.1f}")
            print(f"Median - Expected: {self.inter_eye_median:.1f}, Actual: {np.median(intervals_array):.1f}")
            print(f"Std Dev - Expected: {self.inter_eye_std:.1f}, Actual: {intervals_array.std():.1f}")
        
        return validation_results
    
    def get_summary_statistics(self) -> Dict:
        """Get summary of all generator parameters"""
        return {
            'hrg_distribution': self.hrg_distribution,
            'priority_distribution': self.priority_distribution,
            'bilateral_rate': self.bilateral_rate,
            'inter_eye_statistics': {
                'mean_days': self.inter_eye_mean,
                'median_days': self.inter_eye_median,
                'std_days': self.inter_eye_std
            },
            'inter_eye_lognormal_params': self.inter_eye_params,
            'complexity_levels': {hrg: info['level'] for hrg, info in self.complexity_mapping.items()},
            'base_durations': {hrg: info['base_duration'] for hrg, info in self.complexity_mapping.items()}
        }

if __name__ == "__main__":
    
    # Initialize generator
    generator = CataractPatientGenerator(random_seed=42)
    generator.validate_distributions()
    