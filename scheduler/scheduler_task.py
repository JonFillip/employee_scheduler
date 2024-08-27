from ortools.sat.python import cp_model
import pandas as pd

class NurseScheduler:
    """
    Descr: This class encapsulates all the functionality needed to create, solve, and analyze
    a nurse scheduling problem. It uses Google's OR-Tools library for constraint solving.

    Attributes:
        nurses_df (pd.DataFrame): DataFrame containing nurse information.
        all_nurses (list): List of all nurse identifiers.
        num_nurses (int): Total number of nurses.
        num_shifts (int): Number of shifts per day.
        num_days (int): Number of days in the scheduling period.
        all_days (range): Range object representing all days.
        all_shifts (range): Range object representing all shifts.
        model (cp_model.CpModel): The constraint programming model.
        shifts (dict): Dictionary to store shift variables.
    """
    def __init__(self, nurses_file, num_shifts, num_days):
        """
        Initializes the NurseScheduler with the given parameters.

        Args:
            nurses_file (str): Path to the CSV file containing nurse information.
            num_shifts (int): Number of shifts per day.
            num_days (int): Number of days in the scheduling period.
        """
        self.nurses_df = pd.read_csv(nurses_file)
        self.all_nurses = self.nurses_df['Nurse'].tolist()
        self.num_nurses = len(self.all_nurses)
        self.num_shifts = num_shifts
        self.num_days = num_days
        self.all_days = range(self.num_days)
        self.all_shifts = range(self.num_shifts)
        
        self.model = cp_model.CpModel()
        self.shifts = {}
        self._create_variables()

    def _create_variables(self):
        """
        Create binary variables for each nurse-day-shift combination.

        Descr: This method initializes the 'shifts' dictionary with binary variables
        representing whether a nurse works a particular shift on a particular day.
        """
        for d in self.all_days:
            for n in self.all_nurses:
                for s in self.all_shifts:
                    self.shifts[(n, d, s)] = self.model.NewBoolVar(f'shift_n{n}d{d}s{s}')

    def add_basic_constraints(self):
        """
        Add the basic constraints to the model.

        Descr: This method calls helper methods to add the following constraints:
        1. Assign one nurse per shift
        2. Limit each nurse to one shift per day
        3. Distribute shifts evenly among nurses
        """
        self._assign_one_nurse_per_shift()
        self._limit_one_shift_per_day()
        self._distribute_shifts_evenly()

    def _assign_one_nurse_per_shift(self):
        """
        Ensures that each shift on each day is assigned to exactly one nurse.
        """
        for d in self.all_days:
            for s in self.all_shifts:
                self.model.Add(sum(self.shifts[(n, d, s)] for n in self.all_nurses) == 1)

    def _limit_one_shift_per_day(self):
        """
        Ensures that each nurse works at most one shift per day.
        """
        for d in self.all_days:
            for n in self.all_nurses:
                self.model.Add(sum(self.shifts[(n, d, s)] for s in self.all_shifts) <= 1)

    def _distribute_shifts_evenly(self):
        """
        Distribute shifts evenly among nurses.

        Descr: The method calculates the minimum and maximum number of shifts each nurse
        should work over the entire scheduling period and adds constraints to enforce this.
        """
        min_shifts_per_nurse = (self.num_shifts * self.num_days) // self.num_nurses
        max_shifts_per_nurse = min_shifts_per_nurse + 1

        for n in self.all_nurses:
            num_shifts_worked = sum(self.shifts[(n, d, s)] for d in self.all_days for s in self.all_shifts)
            self.model.Add(num_shifts_worked >= min_shifts_per_nurse)
            self.model.Add(num_shifts_worked <= max_shifts_per_nurse)

    def distribute_weekend_shifts_evenly(self):
        """
        Distribute weekend shifts evenly among nurses.

        Descr: This method calculates the minimum and maximum number of weekend shifts each nurse
        should work and adds constraints to enforce this.
        """
        weekend_days = [d for d in self.all_days if d % 7 in [5, 6]]
        min_weekend_shifts = (len(weekend_days) * self.num_shifts) // self.num_nurses
        max_weekend_shifts = min_weekend_shifts + 1

        for n in self.all_nurses:
            weekend_shifts = sum(self.shifts[(n, d, s)] for d in weekend_days for s in self.all_shifts)
            self.model.Add(weekend_shifts >= min_weekend_shifts)
            self.model.Add(weekend_shifts <= max_weekend_shifts)

    def prevent_consecutive_shifts(self, shift_set):
        """
        Prevent nurses from working consecutive days for specified shifts.

        Args:
            shift_set (list): List of shift indices to apply this constraint to.
        """
        for n in self.all_nurses:
            for d in range(self.num_days - 1):
                self.model.Add(sum(self.shifts[(n, d, s)] for s in shift_set) + 
                            sum(self.shifts[(n, d + 1, s)] for s in shift_set) <= 1)

    def enforce_consecutive_days_for_shift(self, shift, days):
        """
        Enforce that if a nurse works a specific shift on a certain day of the week,
        they must work the same shift on the specified consecutive days.

        Args:
            shift (int): The shift index to apply this constraint to.
            days (list): List of consecutive day indices within a week (0-6).
        """
        for week in range(self.num_days // 7):
            start_day = week * 7
            for n in self.all_nurses:
                for d in range(len(days) - 1):
                    self.model.Add(self.shifts[(n, start_day + days[d], shift)] == 
                                self.shifts[(n, start_day + days[d+1], shift)])

    def limit_shifts_per_week(self, max_shifts):
        """
        Limit the number of shifts a nurse can work per week.

        Args:
            max_shifts (int): Maximum number of shifts allowed per week.
        """
        for week in range(self.num_days // 7):
            for n in self.all_nurses:
                week_shifts = sum(self.shifts[(n, d, s)] for d in range(week * 7, (week + 1) * 7) 
                                for s in self.all_shifts)
                self.model.Add(week_shifts <= max_shifts)

    def restrict_shift_for_team(self, team, shift):
        """
        Prevent nurses from a specific team from working a particular shift.

        Args:
            team (str): The team identifier.
            shift (int): The shift index to restrict.
        """
        team_nurses = self.nurses_df[self.nurses_df['Team'] == team]['Nurse'].tolist()
        for n in team_nurses:
            for d in self.all_days:
                self.model.Add(self.shifts[(n, d, shift)] == 0)

    def limit_nurses_per_team_per_day(self, max_nurses):
        """
        Limit the number of nurses from each team working on any given day.

        Args:
            max_nurses (int): Maximum number of nurses from a team allowed to work on the same day.
        """
        teams = self.nurses_df['Team'].unique()
        for team in teams:
            team_nurses = self.nurses_df[self.nurses_df['Team'] == team]['Nurse'].tolist()
            for d in self.all_days:
                self.model.Add(sum(self.shifts[(n, d, s)] for n in team_nurses 
                                for s in self.all_shifts) <= max_nurses)

    def balance_shift_types(self):
        """
        Creates variables to measure the imbalance in shift types

        Returns:
            Int: An integer containing the sum of total violations in the shift balances.
        """
        target_shifts_per_type = self.num_days // self.num_shifts
        imbalance = {}
        for n in self.all_nurses:
            for s in self.all_shifts:
                num_shifts_of_type = sum(self.shifts[(n, d, s)] for d in self.all_days)
                imbalance[(n, s)] = self.model.NewIntVar(0, self.num_days, f'imbalance_n{n}_s{s}')
                self.model.Add(imbalance[(n, s)] >= num_shifts_of_type - target_shifts_per_type)
                self.model.Add(imbalance[(n, s)] >= target_shifts_per_type - num_shifts_of_type)

        total_imbalance = self.model.NewIntVar(0, self.num_nurses * self.num_shifts * self.num_days, 'total_imbalance')
        self.model.Add(total_imbalance == sum(imbalance.values()))
        return total_imbalance

    def ensure_minimum_rest_period(self):
        """
        Counts violations of minimum rest period between shifts, especially after night shifts

        Returns:
            Int: An interger suming the total rest violations
        """
        night_shift = 2
        min_rest_days = 2
        rest_violations = []
        for n in self.all_nurses:
            for d in range(self.num_days - min_rest_days):
                # Checks if a nurse works a night shift followed by any shift within the min_rest_days
                violation = self.model.NewBoolVar(f'rest_violation_n{n}_d{d}')
                night_shift_worked = self.shifts[(n, d, night_shift)]
                subsequent_shifts = [self.shifts[(n, d+i, s)] for i in range(1, min_rest_days+1) for s in self.all_shifts]
                self.model.Add(sum([night_shift_worked] + subsequent_shifts) <= 1).OnlyEnforceIf(violation.Not())
                self.model.Add(sum([night_shift_worked] + subsequent_shifts) > 1).OnlyEnforceIf(violation)
                rest_violations.append(violation)

        total_violations = self.model.NewIntVar(0, len(rest_violations), 'total_rest_violations')
        self.model.Add(total_violations == sum(rest_violations))
        return total_violations

    def limit_night_shifts(self):
        """
        Encourages limiting the number of night shifts a nurse can work in a given period

        Returns:
            Int: An integer of the sum of times nurses worked more than permitted number of consecutive night shifts
        """
        night_shift = 2 # Represent shift_id 2 which is the ED Night shift
        target_night_shifts = self.num_days // (2 * self.num_nurses)
        night_shift_violations = []
        for n in self.all_nurses:
            nurse_night_shifts = sum(self.shifts[(n, d, night_shift)] for d in self.all_days)
            violation = self.model.NewIntVar(0, self.num_days, f'night_shifts_violation_n{n}')
            self.model.Add(violation >= nurse_night_shifts - target_night_shifts)
            night_shift_violations.append(violation)

        total_violations = self.model.NewIntVar(0, self.num_nurses * self.num_days, 'total_night_shift_violations')
        self.model.Add(total_violations == sum(night_shift_violations))
        return total_violations

    def ensure_skill_mix(self):
        """
        Encourages a mix of nurses from different teams across shifts using soft constraints
        """
        teams = self.nurses_df['Team'].unique()
        good_mix_count = self.model.NewIntVar(0, self.num_days * self.num_shifts, 'good_mix_count')  # Counts the number of shifts with good team mix

        mix_indicators = []
        for d in self.all_days:
            for s in self.all_shifts:
                team_represented = []
                for team in teams:
                    team_nurses = self.nurses_df[self.nurses_df['Team'] == team]['Nurse'].tolist()
                    team_working = self.model.NewBoolVar(f'team_{team}_working_d{d}_{s}')
                    self.model.Add(sum(self.shifts[(n, d, s)] for n in team_nurses) >= 1).OnlyEnforceIf(team_working)
                    self.model.Add(sum(self.shifts[(n, d, s)] for n in team_nurses) == 0).OnlyEnforceIf(team_working.Not())
                    team_represented.append(team_working)

                # Indicator for good mix (at least 2 teams represented)
                good_mix = self.model.NewBoolVar(f'good_mix_d{d}_s{s}')
                self.model.Add(sum(team_represented) >= 2).OnlyEnforceIf(good_mix)
                self.model.Add(sum(team_represented) < 2).OnlyEnforceIf(good_mix.Not())
                mix_indicators.append(good_mix)
                
        # Set the good_mix_count to the number of shifts with good mix
        self.model.Add(good_mix_count == sum(mix_indicators))
        # Maximize the number of shifts with good mix
        self.model.Maximize(good_mix_count)


    def solve(self):
        """
        Solves the constraint satisfaction problem by calling an instance of CpSolver to compute the optimal solution for the schedule model.

        Returns:
            tuple: A tuple containing the solver object and the solution status.
        """
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)
        return solver, status

    def get_solution(self, solver):
        """
        Retrieve the solution if one exists.

        Args:
            solver (cp_model.CpSolver): The solver object after running solve().

        Returns:
            pd.DataFrame or None: A DataFrame representing the schedule if a solution exists, None otherwise.
        """
        if solver.StatusName() == 'OPTIMAL':
            rota_dict = {(n, d, s) for (n, d, s) in self.shifts if solver.Value(self.shifts[(n, d, s)]) == 1}
            solution = pd.DataFrame(rota_dict, columns=['Nurse', 'Day', 'Shift']).sort_values(by=['Day'])
            rota = solution.pivot(index='Day', columns='Shift', values='Nurse')
            return rota
        else:
            return None

    def summarize_shifts(self, solver):
        """
        Summarize the total and weekend shifts for each nurse.

        Args:
            solver (cp_model.CpSolver): The solver object after running solve().

        Returns:
            pd.DataFrame: A DataFrame summarizing total and weekend shifts for each nurse.
        """
        total_shifts = {n: 0 for n in self.all_nurses}
        weekend_shifts = {n: 0 for n in self.all_nurses}
        shift_types = {n: {s: 0 for s in self.all_shifts} for n in self.all_nurses}
        
        for n in self.all_nurses:
            for d in self.all_days:
                for s in self.all_shifts:
                    if solver.Value(self.shifts[(n, d, s)]) == 1:
                        total_shifts[n] += 1
                        shift_types[n][s] += 1
                        if d % 7 in [5, 6]:
                            weekend_shifts[n] += 1
        
        summary = pd.DataFrame({
            'Nurse': self.all_nurses,
            'Total Shifts': [total_shifts[n] for n in self.all_nurses],
            'Weekend Shifts': [weekend_shifts[n] for n in self.all_nurses],
            'ED Day 1 Shifts': [shift_types[n][0] for n in self.all_nurses],
            'ED Day 2 Shifts': [shift_types[n][1] for n in self.all_nurses],
            'ED Night Shifts': [shift_types[n][2] for n in self.all_nurses]
        })
        return summary
