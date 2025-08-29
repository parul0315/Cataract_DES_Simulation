import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, List
import random

class HealthcareArrivalGenerator:
    
    def __init__(self):
        # Parameters from empirical analysis 
        self.day_params = {
            0: {'mean': 14.8, 'std': 9.4},   # Monday
            1: {'mean': 17.3, 'std': 12.1},  # Tuesday
            2: {'mean': 14.8, 'std': 9.9},   # Wednesday
            3: {'mean': 17.8, 'std': 12.1},  # Thursday
            4: {'mean': 15.1, 'std': 10.8},  # Friday
            5: {'mean': 4.5, 'std': 3.3},    # Saturday
            6: {'mean': 1.6, 'std': 0.8}     # Sunday
        }
        # Empirical distribution parameters
        self.weekday_gamma = {'shape': 1.67, 'scale': 9.58}
        self.weekend_gamma = {'shape': 1.58, 'scale': 2.27}
        self.weekday_negbinom = {'n': 2.43, 'p': 0.132}  
        self.weekend_negbinom = {'n': 2.21, 'p': 0.382}
        
        # Autocorrelation parameters
        self.lag1_correlation = 0.530
        self.correlation_strength = 0.8  
        
        # State variables for correlation
        self.previous_day_arrivals = None
        self.previous_day_residual = 0.0
        self.day_history = [] 
        
        # Random state for reproducibility
        self.rng = np.random.RandomState(42)
    
    def set_random_seed(self, seed: int):
        """Set random seed for reproducible results"""
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    def get_expected_daily_arrivals(self, day_of_week: int) -> float:
        """Get expected arrivals for a given day of week (0=Monday, 6=Sunday)"""
        return self.day_params[day_of_week]['mean']
    
    def _gamma_from_mean_std(self, mean: float, std: float) -> Tuple[float, float]:
        """Convert mean/std to gamma distribution parameters (shape, scale)"""
        if mean <= 0 or std <= 0:
            return 1.0, 1.0
        
        # Method of moments for gamma distribution
        shape = (mean / std) ** 2
        scale = std ** 2 / mean
        return shape, scale
    
    def generate_daily_arrivals_simple(self, day_of_week: int) -> int:
        """
        Simple generator using day-specific parameters correctly
        """
        mean_arrivals = self.day_params[day_of_week]['mean']
        std_arrivals = self.day_params[day_of_week]['std']
        
        # Convert to gamma parameters properly
        shape, scale = self._gamma_from_mean_std(mean_arrivals, std_arrivals)
        
        arrivals = self.rng.gamma(shape, scale)
        return max(0, int(round(arrivals)))
    
    def generate_daily_arrivals_stratified(self, day_of_week: int) -> int:
        """
        Stratified generator using separate weekday/weekend distributions
        """
        if day_of_week in [0, 1, 2, 3, 4]:  # Weekdays
            # Use empirical weekday parameters but adjust to specific day
            base_mean = np.mean([self.day_params[i]['mean'] for i in range(5)])
            day_factor = self.day_params[day_of_week]['mean'] / base_mean
            
            arrivals = self.rng.gamma(self.weekday_gamma['shape'], 
                                    self.weekday_gamma['scale'] * day_factor)
        else:  # Weekends
            # Use empirical weekend parameters but adjust to specific day
            weekend_days = [5, 6]
            base_mean = np.mean([self.day_params[i]['mean'] for i in weekend_days])
            day_factor = self.day_params[day_of_week]['mean'] / base_mean
            
            arrivals = self.rng.gamma(self.weekend_gamma['shape'], 
                                    self.weekend_gamma['scale'] * day_factor)
        
        return max(0, int(round(arrivals)))
    
    def generate_daily_arrivals_negbinom(self, day_of_week: int) -> int:
        """
        Negative binomial generator with proper parameters
        """
        if day_of_week in [0, 1, 2, 3, 4]:  # Weekdays
            # Adjust for specific day
            base_mean = np.mean([self.day_params[i]['mean'] for i in range(5)])
            day_factor = self.day_params[day_of_week]['mean'] / base_mean
            
            # Scale the number of successes by day factor
            n_scaled = self.weekday_negbinom['n'] * day_factor
            arrivals = self.rng.negative_binomial(n_scaled, self.weekday_negbinom['p'])
            
        else:  # Weekends
            weekend_days = [5, 6]
            base_mean = np.mean([self.day_params[i]['mean'] for i in weekend_days])
            day_factor = self.day_params[day_of_week]['mean'] / base_mean
            
            n_scaled = self.weekend_negbinom['n'] * day_factor
            arrivals = self.rng.negative_binomial(n_scaled, self.weekend_negbinom['p'])
        
        return max(0, arrivals)
    
    def generate_daily_arrivals_autocorrelated(self, day_of_week: int) -> int:
        """
        BALANCED: Autocorrelation with proper variance control
        """
        # Get baseline expected arrivals for this day
        expected_arrivals = self.day_params[day_of_week]['mean']
        std_arrivals = self.day_params[day_of_week]['std']
        
        # Start with base generation
        base_arrivals = self.generate_daily_arrivals_simple(day_of_week)
        
        # Apply moderate correlation effect as adjustment
        correlation_adjustment = 0.0
        
        if len(self.day_history) > 0:
            # Use moderate correlation strength to avoid over-volatility
            last_residual = self.day_history[-1]['normalized_residual']
            correlation_adjustment = self.lag1_correlation * last_residual * std_arrivals * 0.5
            
            # Dampen adjustment to prevent runaway effects
            max_adjustment = std_arrivals * 0.8  # Limit to 80% of std
            correlation_adjustment = np.clip(correlation_adjustment, 
                                           -max_adjustment, max_adjustment)
        
        # Apply correlation adjustment
        adjusted_arrivals = base_arrivals + correlation_adjustment
        
        # Ensure reasonable bounds
        min_arrivals = max(0, expected_arrivals * 0.2)  # At least 20% of expected
        max_arrivals = expected_arrivals * 3.0  # At most 300% of expected
        final_arrivals = int(round(np.clip(adjusted_arrivals, min_arrivals, max_arrivals)))
        
        # Update history with actual final values
        residual = final_arrivals - expected_arrivals
        normalized_residual = residual / std_arrivals if std_arrivals > 0 else 0.0
        
        # Apply exponential decay to prevent persistent bias
        if len(self.day_history) > 0:
            # Gradually reduce extreme residuals
            normalized_residual = normalized_residual * 0.9
        
        self.day_history.append({
            'arrivals': final_arrivals,
            'expected': expected_arrivals,
            'residual': residual,
            'normalized_residual': normalized_residual,
            'day_of_week': day_of_week
        })
        
        # Limit history size
        if len(self.day_history) > 5:
            self.day_history.pop(0)
        
        # Update legacy variables
        self.previous_day_arrivals = final_arrivals
        self.previous_day_residual = normalized_residual
        
        return final_arrivals
    
    def generate_arrival_times_within_day(self, daily_total: int, 
                                        day_of_week: int, 
                                        business_hours_only: bool = True) -> List[float]:
        """
        Generate specific arrival times within a day 
        """
        if daily_total == 0:
            return []
        
        if business_hours_only and day_of_week < 5:  # Weekdays
            # Business hours: 8 AM to 6 PM (10 hour window)
            start_hour = 8.0
            end_hour = 18.0
            # Peak hours: 9-11 AM and 2-4 PM
            peak_hours = [(9.0, 11.0), (14.0, 16.0)]
        elif day_of_week < 5:  # Weekdays extended
            start_hour = 7.0
            end_hour = 20.0
            peak_hours = [(9.0, 11.0), (14.0, 16.0)]
        else:  # Weekends - more limited
            start_hour = 9.0
            end_hour = 17.0
            peak_hours = [(10.0, 12.0)]  # Single peak on weekends
        
        arrival_times = []
        
        # Generate arrivals with peak hour bias (60% in peak hours)
        peak_arrivals = int(daily_total * 0.6)
        regular_arrivals = daily_total - peak_arrivals
        
        # Peak hour arrivals
        peak_duration = sum([end - start for start, end in peak_hours])
        if peak_duration > 0:
            for _ in range(peak_arrivals):
                # Choose random peak period
                total_weight = sum([end - start for start, end in peak_hours])
                rand_pos = self.rng.uniform(0, total_weight)
                
                cumulative = 0
                for start, end in peak_hours:
                    cumulative += (end - start)
                    if rand_pos <= cumulative:
                        arrival_time = self.rng.uniform(start, end)
                        arrival_times.append(arrival_time)
                        break
        
        # Regular hour arrivals
        for _ in range(regular_arrivals):
            arrival_time = self.rng.uniform(start_hour, end_hour)
            arrival_times.append(arrival_time)
        
        return sorted(arrival_times)
    
    def generate_arrivals_time_series(self, 
                                    start_date: str, 
                                    num_days: int,
                                    method: str = 'autocorrelated') -> pd.DataFrame:
        """
        Generate a complete time series of arrivals for DES simulation
        """
        # Reset state for clean simulation
        self.previous_day_arrivals = None
        self.previous_day_residual = 0.0
        self.day_history = []
        
        results = []
        current_date = pd.to_datetime(start_date)
        
        for day in range(num_days):
            day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
            
            
            if method == 'simple':
                daily_arrivals = self.generate_daily_arrivals_simple(day_of_week)
            elif method == 'stratified':
                daily_arrivals = self.generate_daily_arrivals_stratified(day_of_week)
            elif method == 'negbinom':
                daily_arrivals = self.generate_daily_arrivals_negbinom(day_of_week)
            elif method == 'autocorrelated':
                daily_arrivals = self.generate_daily_arrivals_autocorrelated(day_of_week)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Generate individual arrival times
            arrival_times = self.generate_arrival_times_within_day(daily_arrivals, day_of_week)
            
            # Create records for each arrival
            for arrival_time in arrival_times:
                arrival_datetime = current_date + timedelta(hours=arrival_time)
                results.append({
                    'date': current_date.date(),
                    'day_of_week': day_of_week,
                    'day_name': current_date.strftime('%A'),
                    'arrival_time': arrival_time,
                    'arrival_datetime': arrival_datetime,
                    'daily_total': daily_arrivals,
                    'expected_daily': self.get_expected_daily_arrivals(day_of_week)
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(results)
    
    def validate_generator(self, start_date: str, num_days: int = 365) -> Dict:
        
        # Generate test data
        ts = self.generate_arrivals_time_series(start_date, num_days, 'autocorrelated')
        
        if len(ts) == 0:
            return {'error': 'No data generated'}
        
        # Calculate daily totals
        daily_totals = ts.groupby(['date', 'day_of_week'])['daily_total'].first().reset_index()
        
        validation_results = {}
        
        # Day-of-week accuracy 
        dow_accuracy = {}
        for dow in range(7):
            expected = self.get_expected_daily_arrivals(dow)
            actual_days = daily_totals[daily_totals['day_of_week'] == dow]['daily_total']
            if len(actual_days) > 0:
                actual_mean = actual_days.mean()
                accuracy = abs(actual_mean - expected) / expected
                dow_accuracy[dow] = {
                    'expected': expected,
                    'actual': actual_mean,
                    'accuracy_pct': (1 - accuracy) * 100,
                    'within_30pct': accuracy <= 0.30  
                }
        
        validation_results['day_of_week_accuracy'] = dow_accuracy
        
        # Weekend reduction
        weekday_mean = daily_totals[daily_totals['day_of_week'] < 5]['daily_total'].mean()
        weekend_mean = daily_totals[daily_totals['day_of_week'] >= 5]['daily_total'].mean()
        weekend_reduction = (1 - weekend_mean / weekday_mean) * 100
        
        validation_results['weekend_reduction'] = {
            'actual': weekend_reduction,
            'target': 77,
            'close_to_target': abs(weekend_reduction - 77) <= 15  # More lenient
        }
        
        # Autocorrelation
        daily_counts = daily_totals.sort_values('date')['daily_total'].values
        if len(daily_counts) > 1:
            empirical_corr = np.corrcoef(daily_counts[:-1], daily_counts[1:])[0,1]
            # Handle NaN correlations
            if np.isnan(empirical_corr):
                empirical_corr = 0.0
            validation_results['autocorrelation'] = {
                'actual': empirical_corr,
                'target': self.lag1_correlation,
                'close_to_target': abs(empirical_corr - self.lag1_correlation) <= 0.20 
            }
        
        validations_passed = 0
        total_validations = 0
        
        for dow_val in dow_accuracy.values():
            total_validations += 1
            if dow_val['within_30pct']:  
                validations_passed += 1
        
        if validation_results['weekend_reduction']['close_to_target']:
            validations_passed += 1
        total_validations += 1
        
        if 'autocorrelation' in validation_results and validation_results['autocorrelation']['close_to_target']:
            validations_passed += 1
        total_validations += 1
        
        validation_results['overall_score'] = (validations_passed / total_validations) * 100
        validation_results['grade'] = 'A' if validation_results['overall_score'] >= 90 else \
                                    'B' if validation_results['overall_score'] >= 75 else \
                                    'C' if validation_results['overall_score'] >= 60 else 'D'
        
        return validation_results
    
    def compare_methods(self, start_date: str, num_days: int = 365) -> pd.DataFrame:
        """
        Compare different generation methods with validation
        """
        methods = ['simple', 'stratified', 'negbinom', 'autocorrelated']
        comparison_results = []
        
        for method in methods:
            try:
                # Generate time series
                ts = self.generate_arrivals_time_series(start_date, num_days, method)
                
                if len(ts) > 0:
                    # Calculate daily totals
                    daily_totals = ts.groupby(['date', 'day_of_week'])['daily_total'].first().reset_index()
                    
                    # Calculate statistics
                    stats_dict = {
                        'method': method,
                        'mean_daily': daily_totals['daily_total'].mean(),
                        'std_daily': daily_totals['daily_total'].std(),
                        'cv': daily_totals['daily_total'].std() / daily_totals['daily_total'].mean(),
                        'total_arrivals': len(ts),
                        'min_daily': daily_totals['daily_total'].min(),
                        'max_daily': daily_totals['daily_total'].max()
                    }
                    
                    # Day-of-week breakdown with proper aggregation
                    dow_stats = daily_totals.groupby('day_of_week')['daily_total'].mean()
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    
                    for dow in range(7):
                        if dow in dow_stats.index:
                            stats_dict[f'{day_names[dow]}_mean'] = dow_stats[dow]
                        else:
                            stats_dict[f'{day_names[dow]}_mean'] = 0.0
                    
                    comparison_results.append(stats_dict)
                    
            except Exception as e:
                print(f"Error with method {method}: {e}")
                continue
        
        return pd.DataFrame(comparison_results)


if __name__ == "__main__":
    # Initialize generator
    generator = HealthcareArrivalGenerator()
    generator.set_random_seed(42)
    generator.validate_generator('2023-04-01', 365)
    
    
