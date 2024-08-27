import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class ScheduleReporter:
    """
    A class for reporting fairness, work pattern, team depletion metrics from the Nurse scheduler
    """

    @staticmethod
    def generate_fairness_report(summary):
        """
        Generates a report on the fairness of shift distribution.

        Args:
            summary (pd.DataFrame): Summary of nurse schedules.

        Returns:
            pd.DataFrame: Fairness report with statistics on shift distribution.
        """
        fairness_report = {
            'Total Shifts': summary['Total Shifts'].agg(['mean', 'std', 'min', 'max']).to_dict(),
            'Weekend Shifts': summary['Weekend Shifts'].agg(['mean', 'std', 'min', 'max']).to_dict(),
            'ED Day 1 Shifts': summary['ED Day 1 Shifts'].agg(['mean', 'std', 'min', 'max']).to_dict(),
            'ED Day 2 Shifts': summary['ED Day 2 Shifts'].agg(['mean', 'std', 'min', 'max']).to_dict(),
            'ED Night Shifts': summary['ED Night Shifts'].agg(['mean', 'std', 'min', 'max']).to_dict(),
        }
        return pd.DataFrame(fairness_report)

    @staticmethod
    def generate_work_pattern_report(rota):
        """
        Generates a report on work patterns.

        Args:
            rota (pd.DataFrame): The full nurse schedule.

        Returns:
            pd.DataFrame: Work pattern report with statistics on consecutive shifts.
        """
        consecutive_shifts = []
        for nurse in rota.columns:
            shift_count = 0
            for day in rota.index:
                if pd.notna(rota.loc[day, nurse]) and rota.loc[day, nurse] != -1:
                    shift_count += 1
                else:
                    if shift_count > 1:
                        consecutive_shifts.append(shift_count)
                    shift_count = 0
            if shift_count > 1:
                consecutive_shifts.append(shift_count)
        
        work_pattern_report = {
            'Metric': ['Mean Consecutive Shifts', 'Max Consecutive Shifts', 'Min Consecutive Shifts',
                    'No more than 4 shifts/week', 'Mon-Wed night shift pattern', 
                    'Minimum rest period', 'Maximum night shifts'],
            'Value': [f"{np.mean(consecutive_shifts):.2f}" if consecutive_shifts else "N/A", 
                    f"{max(consecutive_shifts)}" if consecutive_shifts else "N/A", 
                    f"{min(consecutive_shifts)}" if consecutive_shifts else "N/A",
                    'Enforced by constraint', 
                    'Enforced by constraint',
                    'Enforced by constraint', 
                    'Enforced by constraint']
        }
        return pd.DataFrame(work_pattern_report)

    @staticmethod
    def generate_team_depletion_report(summary, nurse_df):
        """
        Generates a report on team depletion avoidance and shift distribution among teams.

        Args:
            summary (pd.DataFrame): Summary of nurse schedules.
            nurse_df (pd.DataFrame): DataFrame containing nurse team information.

        Returns:
            tuple: A tuple containing team shifts DataFrame and skill mix report DataFrame.
        """
        team_summary = summary.merge(nurse_df, on='Nurse')
        
        team_shifts = team_summary.groupby('Team').agg({
            'Total Shifts': ['sum', 'mean', 'std', 'min', 'max'],
            'ED Day 1 Shifts': ['sum', 'mean'],
            'ED Day 2 Shifts': ['sum', 'mean'],
            'ED Night Shifts': ['sum', 'mean']
        }).reset_index()
        
        team_shifts.columns = ['_'.join(col).strip() for col in team_shifts.columns.values]
        team_shifts.rename(columns={'Team_': 'Team'}, inplace=True)

        team_shifts = team_shifts.replace({np.nan: 0.00})
        
        team_shifts['Compliance'] = 'Max 1 nurse per team per day (enforced by constraint)'
        
        total_days = summary['Total Shifts'].sum() // len(summary)
        avg_nurses_per_team = team_summary.groupby('Team')['Total Shifts'].sum() / total_days
        
        skill_mix_report = pd.DataFrame({
            'Metric': ['Average number of nurses working per team per day'],
            'Value': [f"{avg_nurses_per_team.mean():.2f}"]
        })
        
        return team_shifts, skill_mix_report

    @staticmethod
    def calculate_avg_teams_per_shift(summary, nurse_df, num_shifts):
        """
        Calculates the average number of teams represented per shift.

        Args:
            summary (pd.DataFrame): Summary of nurse schedules.
            nurse_df (pd.DataFrame): DataFrame containing nurse team information.
            num_shifts (int): Number of shifts.

        Returns:
            float: Average number of teams represented per shift.
        """
        team_summary = summary.merge(nurse_df[['Nurse', 'Team']], left_index=True, right_on='Nurse')
        total_teams_represented = 0
        for shift_type in ['ED Day 1 Shifts', 'ED Day 2 Shifts', 'ED Night Shifts']:
            teams_represented = (team_summary[shift_type] > 0).groupby(team_summary['Team']).any().sum()
            total_teams_represented += teams_represented
        avg_teams_per_shift = total_teams_represented / (3 * num_shifts)  # 3 shift types
        return avg_teams_per_shift

class SchedulePlotter:
    """
    A class for creating various plots of nurse schedules.
    """

    @staticmethod
    def plot_shift_summary(summary, output_dir=None):
        """
        Plots a bar chart of total and weekend shifts for each nurse.

        Args:
            summary (pd.DataFrame): Summary of nurse schedules.
            output_dir (str, optional): Directory to save the plot. If None, the plot is displayed instead.
        """
        plt.figure(figsize=(12, 6))
        summary.plot(x='Nurse', y=['Total Shifts', 'Weekend Shifts'], kind='bar')
        plt.title('Shift Distribution Among Nurses')
        plt.xlabel('Nurse')
        plt.ylabel('Number of Shifts')
        plt.legend(title='Shift Type')
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'shift_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_schedule_heatmap(rota, output_dir=None):
        """
        Plots a heatmap of the nurse schedule.

        Args:
            rota (pd.DataFrame): The full nurse schedule.
            output_dir (str, optional): Directory to save the plot. If None, the plot is displayed instead.
        """
        plt.figure(figsize=(20, 10))
        sns.heatmap(rota, cmap='YlGnBu', cbar_kws={'label': 'Nurse'})
        plt.title('Nurse Schedule Heatmap')
        plt.xlabel('Shift')
        plt.ylabel('Day')
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'schedule_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_shift_type_distribution(summary, output_dir=None):
        """
        Plots a stacked bar chart of shift type distribution for each nurse.

        Args:
            summary (pd.DataFrame): Summary of nurse schedules.
            output_dir (str, optional): Directory to save the plot. If None, the plot is displayed instead.
        """
        shift_types = ['ED Day 1 Shifts', 'ED Day 2 Shifts', 'ED Night Shifts']
        plt.figure(figsize=(15, 8))
        summary[shift_types].plot(kind='bar', stacked=True)
        plt.title('Distribution of Shift Types Among Nurses')
        plt.xlabel('Nurse')
        plt.ylabel('Number of Shifts')
        plt.legend(title='Shift Type')
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'shift_type_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
            plt.close()
        else:
            plt.show()

def save_reports(fairness_report, work_pattern_report, team_shifts, skill_mix_report, output_dir):
    """
    Saves all reports to CSV files in the specified output directory.

    Args:
        fairness_report (pd.DataFrame): Fairness report.
        work_pattern_report (pd.DataFrame): Work pattern report.
        team_shifts (pd.DataFrame): Team shifts report.
        skill_mix_report (pd.DataFrame): Skill mix report.
        output_dir (str): Directory to save the reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fairness_report.to_csv(os.path.join(output_dir, f'fairness_report_{timestamp}.csv'))
    work_pattern_report.to_csv(os.path.join(output_dir, f'work_pattern_report_{timestamp}.csv'))
    team_shifts.to_csv(os.path.join(output_dir, f'team_shifts_report_{timestamp}.csv'))
    skill_mix_report.to_csv(os.path.join(output_dir, f'skill_mix_report_{timestamp}.csv'))
