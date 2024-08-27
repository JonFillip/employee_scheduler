from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from scheduler.scheduler_task import NurseScheduler
from scheduler.schedule_reporting import ScheduleReporter, SchedulePlotter

if __name__ == "__main__":
    nurses_filepath = 'data/Nurses.csv'
    scheduler = NurseScheduler(nurses_filepath, num_shifts=3, num_days=56)

    scheduler.add_basic_constraints()
    scheduler.distribute_weekend_shifts_evenly()
    scheduler.prevent_consecutive_shifts([0, 1])
    scheduler.enforce_consecutive_days_for_shift(2, [0, 1, 2])  # Mon, Tue, Wed
    scheduler.limit_shifts_per_week(4)
    scheduler.restrict_shift_for_team('A', 0)
    scheduler.limit_nurses_per_team_per_day(1)
    scheduler.ensure_skill_mix()

    solver, status = scheduler.solve()

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Adding optimization objectives...")
        shift_imbalance = scheduler.balance_shift_types()
        rest_violations = scheduler.ensure_minimum_rest_period()
        night_shift_violations = scheduler.limit_night_shifts()

        # Combine objectives with weights
        scheduler.model.Minimize(shift_imbalance + 10 * rest_violations + 5 * night_shift_violations)

        print("Solving model with optimization objectives...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 600.0  # 10 minutes for final optimization
        status = solver.Solve(scheduler.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            rota = scheduler.get_solution(solver)
            print("\nFinal Rota:")
            display(rota)

            summary = scheduler.summarize_shifts(solver)
            print("\nFinal Shift Summary:")
            display(summary)

            shift_imbalance_value = solver.Value(shift_imbalance)
            rest_violations_value = solver.Value(rest_violations)
            night_shift_violations_value = solver.Value(night_shift_violations)

            print(f"\nShift type imbalance: {shift_imbalance_value}")
            print(f"Rest period violations: {rest_violations_value}")
            print(f"Night shift violations: {night_shift_violations_value}")

            # Create timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Ensure output directory exists
            output_dir = 'output'
            os.makedirs(output_dir, exist_ok=True)

            # Save final rota and summary
            rota.to_csv(f'{output_dir}/final_rota_{timestamp}.csv')
            summary.to_csv(f'{output_dir}/final_summary_{timestamp}.csv')

            # Save violations
            with open(f'{output_dir}/violations_{timestamp}.txt', 'w') as f:
                f.write(f"Shift type imbalance: {shift_imbalance_value}\n")
                f.write(f"Rest period violations: {rest_violations_value}\n")
                f.write(f"Night shift violations: {night_shift_violations_value}\n")

            # Generate and save reports
            fairness_report = ScheduleReporter.generate_fairness_report(summary)
            work_pattern_report = ScheduleReporter.generate_work_pattern_report(rota)
            team_shifts, skill_mix_report = ScheduleReporter.generate_team_depletion_report(summary, scheduler.nurses_df)
            avg_teams_per_shift = ScheduleReporter.calculate_avg_teams_per_shift(summary, scheduler.nurses_df, scheduler.num_shifts)

            # Save reports
            fairness_report.to_csv(f'{output_dir}/fairness_report_{timestamp}.csv')
            work_pattern_report.to_csv(f'{output_dir}/work_pattern_report_{timestamp}.csv')
            team_shifts.to_csv(f'{output_dir}/team_shifts_report_{timestamp}.csv')
            skill_mix_report.to_csv(f'{output_dir}/skill_mix_report_{timestamp}.csv')

            # Generate and save plots
            SchedulePlotter.plot_shift_summary(summary, output_dir)
            SchedulePlotter.plot_schedule_heatmap(rota, output_dir)
            SchedulePlotter.plot_shift_type_distribution(summary, output_dir)

            print(f"\nAll outputs, reports, and plots saved in the '{output_dir}' directory with timestamp {timestamp}")
        else:
            print("Failed to optimize objectives.")
            print(f'Solver status: {solver.StatusName()}')
    else:
        print('ERROR: no solution')