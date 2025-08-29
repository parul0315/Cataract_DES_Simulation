import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class RTTWaitTimeGenerator:
    """
    Generate realistic RTT wait times for cataract DES simulation
    """
    def __init__(self):
        self.overall_stats = {
            'mean': 108.3,
            'median': 58.0,
            'std': 117.5,
            'compliance_18_weeks': 69.4,
            'percentiles': {
                50: 58.0,
                75: 162.0,
                90: 291.0,
                95: 366.0,
                99: 463.0
            }
        }
        
        # Priority
        self.priority_stats = {
            1: {'mean': 107.4, 'median': 57.0, 'std': 117.7, 'compliance': 69.9},
            2: {'mean': 146.6, 'median': 133.0, 'std': 102.0, 'compliance': 47.2},
            3: {'mean': 55.5, 'median': 52.0, 'std': 14.5, 'compliance': 100.0}
        }
        
        # HRG complexity 
        self.hrg_stats = {
            'BZ33Z': {'mean': 195.8, 'median': 195.5, 'std': 135.8, 'volume': 866},
            'BZ30A': {'mean': 163.5, 'median': 132.0, 'std': 129.2, 'volume': 65},
            'BZ32B': {'mean': 157.3, 'median': 139.0, 'std': 122.3, 'volume': 32},
            'BZ32A': {'mean': 154.3, 'median': 185.0, 'std': 128.8, 'volume': 3},
            'BZ30B': {'mean': 153.0, 'median': 125.0, 'std': 111.5, 'volume': 57},
            'BZ31A': {'mean': 115.8, 'median': 70.0, 'std': 115.9, 'volume': 635},
            'BZ34C': {'mean': 110.6, 'median': 63.0, 'std': 117.3, 'volume': 4089},
            'BZ31B': {'mean': 94.5, 'median': 53.5, 'std': 101.1, 'volume': 580},
            'BZ34A': {'mean': 88.7, 'median': 42.5, 'std': 109.9, 'volume': 916},
            'BZ34B': {'mean': 87.0, 'median': 43.0, 'std': 105.5, 'volume': 3379}
        }
        
        # Seasonal multipliers from patient data
        self.seasonal_multipliers = {
            1: 0.74,   
            2: 0.74,   
            3: 0.92,  
            4: 1.20,   
            5: 1.29,  
            6: 1.27,   
            7: 1.09,   
            8: 1.14,  
            9: 1.27,   
            10: 1.02,  
            11: 0.86,  
            12: 0.87  
        }
        
        self._fit_distributions()
        
        # Inter-eye interval for bilateral surgeries
        self.bilateral_interval = {
            'mean': 158.6,
            'median': 117.0,
            'distribution': stats.lognorm(s=0.8, scale=120)  # Approximate fit
        }
    
    def _fit_distributions(self):
        """Fit lognormal distributions to each priority group"""
        self.distributions = {}
        
        for priority, params in self.priority_stats.items():
            mean_days = params['mean']
            std_days = params['std']
            
            
            if mean_days > 0 and std_days > 0:
                # Method of moments for lognormal 
                variance = std_days ** 2
                mu = np.log(mean_days ** 2 / np.sqrt(variance + mean_days ** 2))
                sigma = np.sqrt(np.log(1 + variance / mean_days ** 2))
                
                # limit sigma to prevent extreme values
                sigma = min(sigma, 2.0)
                
                self.distributions[priority] = {
                    'type': 'lognormal',
                    'params': {'s': sigma, 'scale': np.exp(mu)}
                }
            else:
                # Fallback to normal dist.
                self.distributions[priority] = {
                    'type': 'normal',
                    'params': {'loc': mean_days, 'scale': std_days}
                }
    
    def generate_rtt_wait(self, priority=1, hrg=None, referral_month=None, random_state=None):
        """
        Generate a single RTT wait time
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # base wait time from priority distribution
        if priority in self.distributions:
            dist_params = self.distributions[priority]
            
            if dist_params['type'] == 'lognormal':
                wait_time = stats.lognorm.rvs(**dist_params['params'])
            else:
                wait_time = stats.norm.rvs(**dist_params['params'])
        else:
            # priority 1 if unknown
            wait_time = stats.lognorm.rvs(**self.distributions[1]['params'])
        
        # Apply HRG complexity adjustment
        if hrg and hrg in self.hrg_stats:
            base_mean = self.priority_stats.get(priority, self.priority_stats[1])['mean']
            hrg_mean = self.hrg_stats[hrg]['mean']
            complexity_factor = hrg_mean / base_mean
            wait_time *= complexity_factor
        
        # Apply seasonal adjustment
        if referral_month and referral_month in self.seasonal_multipliers:
            wait_time *= self.seasonal_multipliers[referral_month]
        
        # Ensure non-negative and realistic bounds
        wait_time = max(0, min(wait_time, 750)) 
        
        return round(wait_time, 1)
    
    def generate_batch_waits(self, n_patients, priorities=None, hrgs=None, 
                           referral_months=None, random_state=None):
        """
        Generate wait times for a batch of patients
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate default priorities if not provided 
        if priorities is None:
            priority_weights = [10368/10622, 250/10622, 4/10622]  
            priorities = np.random.choice([1, 2, 3], size=n_patients, p=priority_weights)
        
        # Generate default months if not provided
        if referral_months is None:
            referral_months = np.random.choice(range(1, 13), size=n_patients)
        
        # Generate default HRGs if not provided
        if hrgs is None:
            hrg_codes = list(self.hrg_stats.keys())
            hrg_weights = [self.hrg_stats[hrg]['volume'] for hrg in hrg_codes]
            hrg_weights = np.array(hrg_weights) / sum(hrg_weights)
            hrgs = np.random.choice(hrg_codes, size=n_patients, p=hrg_weights)
        
        # Generate wait times
        wait_times = []
        for i in range(n_patients):
            priority = priorities[i] if hasattr(priorities, '__iter__') else priorities
            hrg = hrgs[i] if hasattr(hrgs, '__iter__') else hrgs
            month = referral_months[i] if hasattr(referral_months, '__iter__') else referral_months
            
            wait_time = self.generate_rtt_wait(priority, hrg, month)
            wait_times.append(wait_time)
        
        return wait_times
    
    def generate_bilateral_interval(self, random_state=None):
        """Generate time between bilateral surgeries"""
        if random_state is not None:
            np.random.seed(random_state)
        
        interval = self.bilateral_interval['distribution'].rvs()
        return max(30, min(interval, 365))  
    
    def calculate_18_week_compliance(self, wait_times):
        """Calculate 18-week RTT compliance rate"""
        within_18_weeks = sum(1 for wait in wait_times if wait <= 126)
        return (within_18_weeks / len(wait_times)) * 100
    
    def validate_generator(self, n_samples=10000):
        """Validate the generator against historical data"""
        
        # Generate sample data
        sample_waits = self.generate_batch_waits(n_samples, random_state=42)
        
        # statistics of sample wait time
        sample_mean = np.mean(sample_waits)
        sample_median = np.median(sample_waits)
        sample_std = np.std(sample_waits)
        sample_compliance = self.calculate_18_week_compliance(sample_waits)
        
        print(f"Generated vs Historical Comparison ({n_samples:,} samples):")
        print(f"{'Metric':<20} {'Historical':<12} {'Generated':<12} {'Difference':<12}")
        print("-" * 58)
        print(f"{'Mean wait (days)':<20} {self.overall_stats['mean']:<12.1f} {sample_mean:<12.1f} {abs(sample_mean - self.overall_stats['mean']):<12.1f}")
        print(f"{'Median wait (days)':<20} {self.overall_stats['median']:<12.1f} {sample_median:<12.1f} {abs(sample_median - self.overall_stats['median']):<12.1f}")
        print(f"{'Std dev (days)':<20} {self.overall_stats['std']:<12.1f} {sample_std:<12.1f} {abs(sample_std - self.overall_stats['std']):<12.1f}")
        print(f"{'18-week compliance':<20} {self.overall_stats['compliance_18_weeks']:<12.1f} {sample_compliance:<12.1f} {abs(sample_compliance - self.overall_stats['compliance_18_weeks']):<12.1f}")
        
        # Percentile comparison
        print(f"\nPercentile Comparison:")
        for p in [50, 75, 90, 95, 99]:
            sample_percentile = np.percentile(sample_waits, p)
            historical_percentile = self.overall_stats['percentiles'][p]
            print(f"  {p}th percentile: {historical_percentile:.1f} vs {sample_percentile:.1f} (diff: {abs(sample_percentile - historical_percentile):.1f})")
        
        return sample_waits
    
    def get_parameters_summary(self):
        """Return summary of all parameters for documentation"""
        return {
            'overall_stats': self.overall_stats,
            'priority_stats': self.priority_stats,
            'hrg_stats': self.hrg_stats,
            'seasonal_multipliers': self.seasonal_multipliers,
            'bilateral_interval': self.bilateral_interval,
            'distributions': self.distributions
        }

if __name__ == "__main__":
    rtt_generator = RTTWaitTimeGenerator()
    rtt_generator.validate_generator()
    
    