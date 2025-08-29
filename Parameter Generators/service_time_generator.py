import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class CataractServiceTimeGenerator:
    """
    Generate realistic service times for cataract des simulation
    """
    
    def __init__(self):
        # Surgery durations by HRG code - based on NHS data 
        self.hrg_surgery_times = {
            'BZ31A': {'mean': 45, 'std': 12},  # Very Major, Cataract with CC
            'BZ31B': {'mean': 40, 'std': 10},  # Very Major, Cataract without CC  
            'BZ32A': {'mean': 35, 'std': 8},   # Major, Cataract with CC
            'BZ32B': {'mean': 30, 'std': 7},   # Major, Cataract without CC
            'BZ33Z': {'mean': 25, 'std': 6},   # Intermediate, Cataract
            'BZ34A': {'mean': 22, 'std': 5},   # Minor, Complex Cataract
            'BZ34B': {'mean': 18, 'std': 4},   # Minor, Simple Phaco
            'BZ35Z': {'mean': 15, 'std': 3},   # Minor, Very Simple
        }
        
        # Surgery durations by complexity level 
        self.complexity_surgery_times = {
            'very_major': {'mean': 45, 'std': 12},
            'major': {'mean': 32, 'std': 8},
            'intermediate': {'mean': 25, 'std': 6},
            'minor': {'mean': 20, 'std': 5},
            'daycase': {'mean': 18, 'std': 4},
        }
        
        # Activity type modifiers
        self.activity_type_modifiers = {
            'Inpatient': 1.2,      
            'Day Case': 1.0,       # Standard time
            'Inpatient/Day Case': 1.0  # Default
        }
        
        # Pre-operative assessment times (minutes)
        self.preop_assessment_times = {
            'routine': {'mean': 30, 'std': 8},
            'complex': {'mean': 45, 'std': 12},
            'very_complex': {'mean': 60, 'std': 15}
        }
        
        # Post-operative follow-up times (minutes)  
        self.postop_followup_times = {
            'routine_1week': {'mean': 10, 'std': 3},
            'routine_6week': {'mean': 15, 'std': 4},
            'complex_followup': {'mean': 20, 'std': 5}
        }
        
        # Theatre setup/changeover times (minutes)
        self.theatre_times = {
            'setup_time': {'mean': 15, 'std': 3},
            'changeover_time': {'mean': 10, 'std': 2},
            'cleaning_time': {'mean': 8, 'std': 2}
        }
        
        # Administrative processing times (minutes)
        self.admin_times = {
            'booking_appointment': {'mean': 5, 'std': 2},
            'pre_admission': {'mean': 15, 'std': 4},
            'discharge_admin': {'mean': 12, 'std': 3}
        }
        
        # Bounds for realistic times
        self.min_surgery_time = 8   # Min surgery time
        self.max_surgery_time = 120 # Max surgery time
    
    def get_surgery_duration(self, hrg_code=None, complexity=None, activity_type='Day Case'):
        """
        Generate surgery duration based on HRG code, complexity, and activity type
        """
    
        if hrg_code and hrg_code in self.hrg_surgery_times:
            time_params = self.hrg_surgery_times[hrg_code]
      
        elif complexity and complexity in self.complexity_surgery_times:
            time_params = self.complexity_surgery_times[complexity]
        else:
            # Default to daycase 
            time_params = self.complexity_surgery_times['daycase']
        
        base_time = np.random.normal(time_params['mean'], time_params['std'])
        
        # Apply activity type modifier
        modifier = self.activity_type_modifiers.get(activity_type, 1.0)
        surgery_time = base_time * modifier
        
        # Apply bounds
        surgery_time = max(self.min_surgery_time, min(self.max_surgery_time, surgery_time))
        
        return round(surgery_time, 1)
    
    def get_preop_assessment_duration(self, complexity='routine'):
        """Generate pre-operative assessment duration"""
        
        if complexity in ['very_major', 'major']:
            assessment_type = 'very_complex'
        elif complexity in ['intermediate']:
            assessment_type = 'complex'  
        else:
            assessment_type = 'routine'
        
        time_params = self.preop_assessment_times[assessment_type]
        duration = max(5, np.random.normal(time_params['mean'], time_params['std']))
        
        return round(duration, 1)
    
    def get_postop_followup_duration(self, followup_type='routine_6week'):
        """Generat post-op  follow-up duration"""
        
        time_params = self.postop_followup_times.get(followup_type, 
                                                   self.postop_followup_times['routine_6week'])
        duration = max(5, np.random.normal(time_params['mean'], time_params['std']))
        
        return round(duration, 1)
    
    def get_theatre_overhead_time(self, overhead_type='setup_time'):
        """Generate theatre setup/changeover times"""
        
        time_params = self.theatre_times.get(overhead_type, 
                                           self.theatre_times['setup_time'])
        duration = max(1, np.random.normal(time_params['mean'], time_params['std']))
        
        return round(duration, 1)
    
    def get_admin_processing_time(self, admin_type='booking_appointment'):
        """Generate administration times"""
        
        time_params = self.admin_times.get(admin_type, 
                                         self.admin_times['booking_appointment'])
        duration = max(1, np.random.normal(time_params['mean'], time_params['std']))
        
        return round(duration, 1)
    
    def generate_complete_episode_times(self, hrg_code=None, complexity=None, 
                                      activity_type='Day Case'):
        """
        Generate all time components for a complete cataract episode
        """
        
        episode_times = {
            'surgery_duration': self.get_surgery_duration(hrg_code, complexity, activity_type),
            'preop_assessment': self.get_preop_assessment_duration(complexity),
            'postop_1week': self.get_postop_followup_duration('routine_1week'),
            'postop_6week': self.get_postop_followup_duration('routine_6week'),
            'theatre_setup': self.get_theatre_overhead_time('setup_time'),
            'theatre_changeover': self.get_theatre_overhead_time('changeover_time'),
            'booking_admin': self.get_admin_processing_time('booking_appointment'),
            'pre_admission': self.get_admin_processing_time('pre_admission'),
            'discharge_admin': self.get_admin_processing_time('discharge_admin')
        }
        
        # Calculate total theatre time (surgery + setup + changeover)
        episode_times['total_theatre_time'] = (episode_times['surgery_duration'] + 
                                             episode_times['theatre_setup'] + 
                                             episode_times['theatre_changeover'])
        
        # Calculate total episode time (excluding waiting times)
        episode_times['total_episode_time'] = sum([
            episode_times['surgery_duration'],
            episode_times['preop_assessment'], 
            episode_times['postop_1week'],
            episode_times['postop_6week'],
            episode_times['booking_admin'],
            episode_times['pre_admission'],
            episode_times['discharge_admin']
        ])
        
        return episode_times
    
    def validate_generator(self, n_simulations=1000):
        """Validate the service time generator"""
        
        # Test HRG-based surgery times
        print("\nSurgery Duration by HRG Code:")
        
        test_hrgs = ['BZ34B', 'BZ33Z', 'BZ31A']  
        
        for hrg in test_hrgs:
            samples = [self.get_surgery_duration(hrg_code=hrg) for _ in range(n_simulations)]
            expected_mean = self.hrg_surgery_times[hrg]['mean']
            expected_std = self.hrg_surgery_times[hrg]['std']
            simulated_mean = np.mean(samples)
            simulated_std = np.std(samples)
            
            print(f"{hrg}: Expected={expected_mean:4.1f}±{expected_std:4.1f}min, "
                  f"Simulated={simulated_mean:4.1f}±{simulated_std:4.1f}min")
        
        # Test complexity-based surgery times  
        print("\nSurgery Duration by Complexity:")
        
        test_complexities = ['daycase', 'intermediate', 'very_major']
        
        for complexity in test_complexities:
            samples = [self.get_surgery_duration(complexity=complexity) for _ in range(n_simulations)]
            expected_mean = self.complexity_surgery_times[complexity]['mean']
            expected_std = self.complexity_surgery_times[complexity]['std']
            simulated_mean = np.mean(samples)
            simulated_std = np.std(samples)
            
            print(f"{complexity:>12}: Expected={expected_mean:4.1f}±{expected_std:4.1f}min, "
                  f"Simulated={simulated_mean:4.1f}±{simulated_std:4.1f}min")
        
        # Test activity type modifiers
        print("\nActivity Type Modifiers:")
        
        daycase_times = [self.get_surgery_duration('BZ34B', activity_type='Day Case') 
                        for _ in range(n_simulations)]
        inpatient_times = [self.get_surgery_duration('BZ34B', activity_type='Inpatient') 
                          for _ in range(n_simulations)]
        
        daycase_mean = np.mean(daycase_times)
        inpatient_mean = np.mean(inpatient_times)
        modifier_actual = inpatient_mean / daycase_mean
        modifier_expected = self.activity_type_modifiers['Inpatient']
        
        print(f"Day Case (BZ34B): {daycase_mean:.1f} min")
        print(f"Inpatient (BZ34B): {inpatient_mean:.1f} min")
        print(f"Modifier: Expected={modifier_expected:.2f}, Actual={modifier_actual:.2f}")
        
        print("\nComplete Episode Times (Sample):")

        
        for i in range(3):
            episode = self.generate_complete_episode_times('BZ34B', 'daycase')
            print(f"Episode {i+1}:")
            print(f"  Surgery: {episode['surgery_duration']:.1f}min")
            print(f"  Total Theatre: {episode['total_theatre_time']:.1f}min") 
            print(f"  Total Episode: {episode['total_episode_time']:.1f}min")
            print()
    
    def plot_validation(self, n_simulations=1000):
        """Create validation plots for service times"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Service Time Generator Validation', fontsize=16)
        
        # 1. Surgery times by HRG
        hrg_codes = ['BZ34B', 'BZ33Z', 'BZ31A']
        hrg_labels = ['Simple\n(BZ34B)', 'Intermediate\n(BZ33Z)', 'Complex\n(BZ31A)']
        
        surgery_times_data = []
        for hrg in hrg_codes:
            times = [self.get_surgery_duration(hrg_code=hrg) for _ in range(n_simulations)]
            surgery_times_data.append(times)
        
        axes[0,0].boxplot(surgery_times_data, labels=hrg_labels)
        axes[0,0].set_title('Surgery Duration by HRG Code')
        axes[0,0].set_ylabel('Duration (minutes)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Activity type comparison
        daycase_times = [self.get_surgery_duration('BZ34B', activity_type='Day Case') 
                        for _ in range(n_simulations)]
        inpatient_times = [self.get_surgery_duration('BZ34B', activity_type='Inpatient') 
                          for _ in range(n_simulations)]
        
        axes[0,1].hist(daycase_times, bins=30, alpha=0.7, label='Day Case', density=True)
        axes[0,1].hist(inpatient_times, bins=30, alpha=0.7, label='Inpatient', density=True)
        axes[0,1].set_title('Surgery Duration: Day Case vs Inpatient (BZ34B)')
        axes[0,1].set_xlabel('Duration (minutes)')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Complete episode breakdown
        episodes = [self.generate_complete_episode_times('BZ34B', 'daycase') 
                   for _ in range(n_simulations)]
        
        components = ['surgery_duration', 'preop_assessment', 'postop_6week', 
                     'theatre_setup', 'pre_admission']
        component_labels = ['Surgery', 'Pre-op\nAssess', 'Post-op\n6wk', 
                           'Theatre\nSetup', 'Pre-admission']
        
        component_data = []
        for component in components:
            component_data.append([ep[component] for ep in episodes])
        
        axes[1,0].boxplot(component_data, labels=component_labels)
        axes[1,0].set_title('Episode Component Durations (BZ34B Daycase)')
        axes[1,0].set_ylabel('Duration (minutes)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Total theatre time distribution
        theatre_times = [ep['total_theatre_time'] for ep in episodes]
        episode_times = [ep['total_episode_time'] for ep in episodes]
        
        axes[1,1].hist(theatre_times, bins=30, alpha=0.7, label='Theatre Time', density=True)
        axes[1,1].hist(episode_times, bins=30, alpha=0.7, label='Total Episode', density=True)
        axes[1,1].set_title('Total Times Distribution (BZ34B Daycase)')
        axes[1,1].set_xlabel('Duration (minutes)')
        axes[1,1].set_ylabel('Density')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_times_for_dataframe(self, df, hrg_col='HRG_y', complexity_col='Complexity_y', 
                                   activity_type_col='Activity Type_y'):
        """
        Generate service times 
        """
        
        df_with_times = df.copy()
        
        # Generate surgery durations
        surgery_times = []
        preop_times = []
        postop_times = []
        theatre_times = []
        
        for _, row in df.iterrows():
            hrg = row.get(hrg_col)
            complexity = row.get(complexity_col)
            activity_type = row.get(activity_type_col, 'Day Case')
            
            # Generate times
            surgery_time = self.get_surgery_duration(hrg, complexity, activity_type)
            preop_time = self.get_preop_assessment_duration(complexity)
            postop_time = self.get_postop_followup_duration()
            theatre_time = surgery_time + self.get_theatre_overhead_time()
            
            surgery_times.append(surgery_time)
            preop_times.append(preop_time)
            postop_times.append(postop_time)
            theatre_times.append(theatre_time)
      
        df_with_times['surgery_duration_min'] = surgery_times
        df_with_times['preop_assessment_min'] = preop_times
        df_with_times['postop_followup_min'] = postop_times
        df_with_times['total_theatre_time_min'] = theatre_times
        
        return df_with_times

if __name__ == "__main__":
    generator = CataractServiceTimeGenerator()
    generator.validate_generator()
   