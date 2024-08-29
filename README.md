# My Approach to Solving the Nurse Scheduling Problem

## Problem Overview
The nurse scheduling problem involves creating a work schedule for a group of nurses over a specified period, subject to various constraints. This project was completed in two parts, with the second part focusing on additional optimizations and comprehensive reporting.

## My Solution Approach

1. **Problem Modeling**
   - First, I define the problem variables: nurses, shifts, and days.
   - I identify all the constraints, such as shift coverage, nurse workload, and team restrictions.
   - My objective is to create a feasible schedule that satisfies all these constraints.

2. **Constraint Programming with OR-Tools**
   - The brief of the task required the use of Google's OR-Tools library, specifically the CP-SAT (Constraint Programming - Satisfiability) solver.
   - I create binary variables for each nurse-day-shift combination.
   - I express all constraints mathematically using these variables.

3. **Key Constraints Implementation**
   - I ensure each shift is covered by exactly one nurse.
   - I limit each nurse to one shift per day.
   - I distribute shifts evenly among nurses.
   - I handle weekend shifts fairly.
   - I implement team-specific and consecutive shift restrictions.

4. **Solving the Model**
   - I use the CP-SAT solver to find a solution that satisfies all constraints.
   - The solver employs advanced algorithms to efficiently search the solution space.

5. **Solution Analysis and Visualization**
   - Once I have a solution, I extract it from the solver.
   - I create a readable schedule format (e.g., a pandas DataFrame).
   - I generate summary statistics (e.g., shifts per nurse, weekend distribution).
   - I visualize the schedule and statistics for easy interpretation (See: employee_scheduler.ipynb).

6. **Iterative Refinement**
   - If needed, I adjust constraints or add new ones based on the results.
   - I re-solve and analyze until I obtain a satisfactory schedule.

7. **Additional Optimizations (Part 2)**
   - I implement additional rules to further improve fairness, work patterns, and team distribution.
   - I create a more comprehensive reporting system to evaluate the schedule quality.

8. **Extended Reporting and Visualization (Part 2)**
   - I generate detailed reports on fairness, work patterns, and team depletion.
   - I create visualizations to better understand the schedule distribution.
   - All reports and visualizations are automatically saved for future reference.

## Project File Structure
```
medmodus_task/
├── data/
│   ├── Nurses.csv
├── output/
├── scheduler/
│   ├── __init__.py
│   ├── scheduler_task.py
│   ├── schedule_reporting.py
│   ├── main.py
├── employee_scheduler.ipynb
├── README.md
├── requirement.txt
```


## Key Components

- `scheduler_task.py`: Contains the `NurseScheduler` class which handles the core scheduling logic.
- `schedule_reporting.py`: Includes `ScheduleReporter` and `SchedulePlotter` classes for generating reports and visualizations.
- `main.py`: The main script that runs the entire scheduling process.
- `employee_scheduler.ipynb`: A Jupyter notebook for interactive exploration of the scheduling results.

## How to Run

1. Ensure all requirements are installed: `pip install -r requirements.txt`
2. Run the main script: `python -m scheduler.main`
3. Check the `output/` directory for generated reports and visualizations.

## Conclusion

This approach leverages the power of constraint programming to handle the complex interplay of various scheduling rules. It allows for flexibility in adding or modifying constraints and can efficiently handle large problem instances that would be impractical to solve manually. The addition of comprehensive reporting and visualization in the second part of the project provides valuable insights into the quality and fairness of the generated schedules.
