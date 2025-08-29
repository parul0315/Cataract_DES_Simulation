import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class CataractCapacityGenerator:
    """
    Generate daily surgery capacity based on cataract service data 
    """
    
    def __init__(self):
        
        self.overall_mu = 23.8
        self.overall_sigma = 24.7
        
        
        weekday_volumes = {
            'Monday': 22.6,
            'Tuesday': 26.2, 
            'Wednesday': 28.1,
            'Thursday': 25.7,
            'Friday': 25.4,
            'Saturday': 10.9,
            'Sunday': 12.1
        }
        
       
        self.weekday_means = {
            0: weekday_volumes['Monday'],   
            1: weekday_volumes['Tuesday'],   
            2: weekday_volumes['Wednesday'],
            3: weekday_volumes['Thursday'],  
            4: weekday_volumes['Friday'],    
            5: weekday_volumes['Saturday'],  
            6: weekday_volumes['Sunday'],   
        }
        
        # Calculate weekday factors 
        self.weekday_factors = {
            weekday: mean / self.overall_mu 
            for weekday, mean in self.weekday_means.items()
        }
        
        # Monthly multipliers from seasonal analysis
        self.monthly_multipliers = {
            1: 1.77,   
            2: 1.44,   
            3: 1.36,   
            4: 0.75,   
            5: 0.71,   
            6: 0.75,   
            7: 0.94,   
            8: 0.94,   
            9: 0.91,   
            10: 0.98, 
            11: 1.29, 
            12: 1.15   
        }
        
        # Complexity distribution 
        self.complexity_distribution = {
            'daycase': 0.784,
            'intermediate': 0.113,
            'very_major': 0.088,
            'minor': 0.011,
            'major': 0.004
        }
        
        # Capacity bounds
        self.min_capacity = 1
        self.max_capacity = 105  
    
    def generate_capacity(self, weekday, month=None, method='weekday_adjusted'):
        """
        Generate daily surgery capacity
        
        Parameters:
        weekday: 0=Monday, 6=Sunday
        month: 1-12 for seasonal adjustment
        method: 'weekday_adjusted', 'empirical', 'simple_normal'
        
        Returns:
        int: Number of surgery slots available
        """
        
        if method == 'weekday_adjusted':
            # Use overall distribution adjusted by weekday pattern
            base_capacity = max(0, np.random.normal(self.overall_mu, self.overall_sigma))
            weekday_capacity = base_capacity * self.weekday_factors[weekday]
            
            # Apply seasonal adjustment if month provided
            if month is not None:
                seasonal_mult = self.monthly_multipliers.get(month, 1.0)
                weekday_capacity *= seasonal_mult
                
            capacity = max(self.min_capacity, min(self.max_capacity, int(weekday_capacity)))
            return capacity
            
        elif method == 'empirical':
            # Use empirical weekday means with variation
            mean_capacity = self.weekday_means[weekday]
            
            # Apply seasonal adjustment if month provided  
            if month is not None:
                mean_capacity *= self.monthly_multipliers.get(month, 1.0)
                
            # Add variation 
            capacity = max(0, np.random.normal(mean_capacity, mean_capacity * 1.04))
            capacity = max(self.min_capacity, min(self.max_capacity, int(capacity)))
            return capacity
            
        elif method == 'simple_normal':
            # Simple normal distribution 
            capacity = max(0, np.random.normal(self.overall_mu, self.overall_sigma))
            return max(self.min_capacity, min(self.max_capacity, int(capacity)))
    
    def generate_weekly_schedule(self, month=None, method='empirical'):
        """Generate a full week's surgery schedule"""
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        schedule = {}
        
        for weekday in range(7):
            capacity = self.generate_capacity(weekday, month, method)
            schedule[days[weekday]] = capacity
            
        return schedule
    
    def generate_monthly_schedule(self, month, year=2024, method='empirical'):
        """Generate full month schedule with proper dates"""
        import calendar
        
        # Get number of days in month
        days_in_month = calendar.monthrange(year, month)[1]
        schedule = []
        
        for day in range(1, days_in_month + 1):
            date = pd.Timestamp(year, month, day)
            weekday = date.weekday()  # 0=Monday, 6=Sunday
            
            capacity = self.generate_capacity(weekday, month, method)
            
            schedule.append({
                'date': date,
                'weekday': date.strftime('%A'),
                'capacity': capacity
            })
        
        return pd.DataFrame(schedule)
    
    def generate_surgery_complexity(self, total_surgeries):
        """
        Generate surgery complexity mix for given total surgeries
        
        Returns:
        dict: Count by complexity type
        """
        complexity_counts = {}
        remaining = total_surgeries
        
        # Generate counts based on probabilities
        for complexity, prob in self.complexity_distribution.items():
            if complexity == 'daycase':  # Handle largest category last
                continue
            count = np.random.binomial(remaining, prob / (1 - sum(complexity_counts.values()) / total_surgeries))
            complexity_counts[complexity] = min(count, remaining)
            remaining -= count
        
        # Assign remainder to daycase
        complexity_counts['daycase'] = remaining
        
        return complexity_counts
    
    def validate_generator(self, n_simulations=1000):
        """Validate the generator against observed data"""
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        print("\nWeekday Pattern Validation:")
        
        for weekday in range(7):
            # Generate samples
            samples = [self.generate_capacity(weekday, method='empirical') for _ in range(n_simulations)]
            
            observed_mean = self.weekday_means[weekday]
            simulated_mean = np.mean(samples)
            simulated_std = np.std(samples)
            error = abs(observed_mean - simulated_mean)
            
            print(f"{days[weekday]:>9}: Observed={observed_mean:5.1f}, "
                  f"Simulated={simulated_mean:5.1f} ± {simulated_std:4.1f}, "
                  f"Error={error:4.1f}")
        
        # Test seasonal patterns
        print("\nSeasonal Pattern Validation:")
        
        for month in [1, 5, 11]:  # Test high, low, high months
            month_names = {1: 'January', 5: 'May', 11: 'November'}
            samples = [self.generate_capacity(1, month, 'empirical') for _ in range(n_simulations)]  # Tuesday as baseline
            
            baseline_mean = self.weekday_means[1]  # Tuesday baseline
            expected_mean = baseline_mean * self.monthly_multipliers[month]
            simulated_mean = np.mean(samples)
            
            print(f"{month_names[month]:>9}: Expected={expected_mean:5.1f}, "
                  f"Simulated={simulated_mean:5.1f}, "
                  f"Multiplier={simulated_mean/baseline_mean:.2f}")
        
        # Test complexity distribution
        print("\nComplexity Distribution Validation:")
        
        total_test_surgeries = 10000
        complexity_counts = self.generate_surgery_complexity(total_test_surgeries)
        
        for complexity, observed_prop in self.complexity_distribution.items():
            simulated_count = complexity_counts.get(complexity, 0)
            simulated_prop = simulated_count / total_test_surgeries
            error = abs(observed_prop - simulated_prop)
            
            print(f"{complexity:>12}: Observed={observed_prop:.3f}, "
                  f"Simulated={simulated_prop:.3f}, "
                  f"Error={error:.3f}")
        
        # Overall statistics validation
        print("\nOverall Statistics Validation:")
        
        all_samples = [self.generate_capacity(np.random.randint(0, 7), method='empirical') 
                      for _ in range(n_simulations * 7)]
        
        print(f"Overall Mean: Observed={self.overall_mu:.1f}, "
              f"Simulated={np.mean(all_samples):.1f}")
        print(f"Overall Std:  Observed={self.overall_sigma:.1f}, "
              f"Simulated={np.std(all_samples):.1f}")
    
    def plot_validation(self, n_simulations=1000):
        """Create validation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Capacity Generator Validation', fontsize=16)
        
        # 1. Weekday patterns
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        observed_means = [self.weekday_means[i] for i in range(7)]
        simulated_means = []
        simulated_stds = []
        
        for weekday in range(7):
            samples = [self.generate_capacity(weekday, method='empirical') for _ in range(n_simulations)]
            simulated_means.append(np.mean(samples))
            simulated_stds.append(np.std(samples))
        
        x_pos = np.arange(len(days))
        width = 0.35
        
        axes[0,0].bar(x_pos - width/2, observed_means, width, label='Observed', alpha=0.8)
        axes[0,0].bar(x_pos + width/2, simulated_means, width, label='Simulated', alpha=0.8)
        axes[0,0].set_title('Average Capacity by Weekday')
        axes[0,0].set_xlabel('Day of Week')
        axes[0,0].set_ylabel('Average Surgeries')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(days)
        axes[0,0].legend()
        
        # 2. Capacity distribution
        all_samples = [self.generate_capacity(np.random.randint(0, 7), method='empirical') 
                      for _ in range(n_simulations)]
        
        axes[0,1].hist(all_samples, bins=30, alpha=0.7, density=True, label='Simulated')
        axes[0,1].axvline(self.overall_mu, color='red', linestyle='--', 
                         label=f'Observed Mean: {self.overall_mu:.1f}')
        axes[0,1].axvline(np.mean(all_samples), color='blue', linestyle='--', 
                         label=f'Simulated Mean: {np.mean(all_samples):.1f}')
        axes[0,1].set_title('Daily Capacity Distribution')
        axes[0,1].set_xlabel('Daily Surgeries')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()
        
        # 3. Monthly patterns
        months = list(range(1, 13))
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_means = []
        
        baseline_tuesday = self.weekday_means[1] 
        
        for month in months:
            samples = [self.generate_capacity(1, month, 'empirical') for _ in range(n_simulations//4)]
            monthly_means.append(np.mean(samples))
        
        expected_means = [baseline_tuesday * self.monthly_multipliers[m] for m in months]
        
        axes[1,0].plot(months, expected_means, 'o-', label='Expected', linewidth=2, markersize=6)
        axes[1,0].plot(months, monthly_means, 's-', label='Simulated', linewidth=2, markersize=6)
        axes[1,0].set_title('Seasonal Pattern (Tuesday Baseline)')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Average Surgeries')
        axes[1,0].set_xticks(months)
        axes[1,0].set_xticklabels(month_names)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Complexity distribution
        complexity_counts = self.generate_surgery_complexity(10000)
        complexities = list(self.complexity_distribution.keys())
        observed_props = [self.complexity_distribution[c] for c in complexities]
        simulated_props = [complexity_counts.get(c, 0) / 10000 for c in complexities]
        
        x_pos = np.arange(len(complexities))
        width = 0.35
        
        axes[1,1].bar(x_pos - width/2, observed_props, width, label='Observed', alpha=0.8)
        axes[1,1].bar(x_pos + width/2, simulated_props, width, label='Simulated', alpha=0.8)
        axes[1,1].set_title('Surgery Complexity Distribution')
        axes[1,1].set_xlabel('Complexity Level')
        axes[1,1].set_ylabel('Proportion')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(complexities, rotation=45)
        axes[1,1].legend()
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    generator = CataractCapacityGenerator()
    generator.validate_generator()