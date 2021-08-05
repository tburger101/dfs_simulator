import multiprocessing as mp
import time,datetime, copy
import pandas as pd, decimal, numpy as np, ast


class lineup_simulator:
    def __init__(self):
        self.salary_cap=None
        self.roster_construction=None
        self.total_roster=None
        self.stack_player_percent=None
        self.opp_stack_percent=None
        self.contest_size=None
        self.sims=None
        self.top_percent=None
        self.ownership=None
        self.csv_path=None
        self.lineups=None
        self.corr=None
        self.player_corr_matrix=None
        self.player_depth=None
        self.player_covariance=None
        self.player_samples=None
        self.sim_ranks=None
        self.sim_ownership=None
        self.opp_teams=None

    def initialize_variables(self):
        with open('config.txt') as f:
            settings_file = f.read()
        settings = ast.literal_eval(settings_file)
        self.salary_cap = settings.get('salary_cap')
        self.roster_construction = settings.get('roster_const')
        self.total_roster=settings.get('total_roster')
        self.stack_player_percent = settings.get('stack_player_percent')
        self.opp_stack_percent = settings.get('opp_stack_percent')
        self.contest_size = settings.get('contest_size')
        self.sims =settings.get('num_sims')
        self.top_percent = settings.get('top_percent')
        self.csv_path = settings.get('path')

        ownership_loc=str(self.csv_path+"/"+"ownership_player.csv")
        corr_loc=str(self.csv_path+"/"+"correlation.csv")
        projection_loc = str(self.csv_path + "/" + "player_projection_std.csv")

        ownership_df=pd.read_csv(ownership_loc)
        projections_df=pd.read_csv(projection_loc)
        ownership_df=pd.merge(ownership_df, projections_df, on='player_name')
        ownership_df.set_index('player_name', inplace=True)
        ownership_df.sort_values(by=['team', 'position', 'fp'], inplace=True, ascending=False)
        self.corr=pd.read_csv(corr_loc)
        self.corr.set_index('Position', inplace=True)
        self.ownership=ownership_df

    def team_creation(self, total_teams):
        total_valid_lineups, invalid_lineup = 0, 0
        ownership_df=self.ownership

        lineup_construction, salary_cap, total_roster, stack_player_percent,opp_stack_percent=self.roster_construction, self.salary_cap, \
                self.total_roster, self.stack_player_percent, self.opp_stack_percent
        ownership_list = []
        final_lineups = []

        for index, row in ownership_df.iterrows():
            player_data = [index, row['team'], row['opp'], row['position'], row['salary'],
                           row['ownership']]
            multiplier = decimal.Decimal(row['ownership']) * 100
            multiplier = int(round(multiplier))
            for x in range(multiplier):
                ownership_list.append(player_data)

        lineup_pos_tracker = {}

        realistic_salary_floor = salary_cap * .992
        min_player_salary = (salary_cap / total_roster) * .54

        total_probability_sample = pd.DataFrame(ownership_list, columns=['player_name', 'team', 'opp', 'position', 'salary', 'ownership'])

        team_df = ownership_df.copy(deep=True)

        while total_valid_lineups < total_teams:
            total_players, total_salary, avg_slary_remain_min = 0,0,0
            avg_salary_remain_max = salary_cap
            sample_df = total_probability_sample.copy(deep=True)
            min_viable_salary = sample_df[sample_df['position'].isin(['WR', 'TE', 'RB'])]['salary'].min()
            max_viable_salary = sample_df[sample_df['position'].isin(['WR', 'TE', 'RB'])]['salary'].max()
            team_df['own'] = 0
            for x in lineup_construction:
                lineup_pos_tracker[x[0]] = x[1]

            for position_player in lineup_construction:
                pos = position_player[0]
                remaining_salary = salary_cap - total_salary
                remaining_players = total_roster - total_players
                if remaining_salary <= 0 or remaining_players == 0:
                    break
                if remaining_players < 4:
                    avg_slary_remain_min = max(((remaining_salary / remaining_players) * .8), min_viable_salary)
                    avg_salary_remain_max = (remaining_salary / remaining_players) * 1.15
                if remaining_players == 2:
                    avg_slary_remain_min = max(remaining_salary - max_viable_salary, min_viable_salary)
                    avg_salary_remain_max = max(remaining_salary - min_viable_salary, min_viable_salary)
                if remaining_players == 1:
                    avg_slary_remain_min = realistic_salary_floor - total_salary
                    avg_salary_remain_max = remaining_salary
                if lineup_pos_tracker.get(pos) == 0:
                    continue

                if pos == 'QB':
                    stack_values = np.random.multinomial(1, stack_player_percent)
                    opp_stack_value = np.random.binomial(1, opp_stack_percent)
                    counter = 0
                    for x in stack_values:
                        if x == 1:
                            qb_stack_players = counter
                            break
                        else:
                            counter = counter + 1

                    player_info = sample_df[(sample_df['position'] == pos)].sample(n=1).values
                    player = player_info[0][0]
                    qb_team = player_info[0][1]
                    qb_opp = player_info[0][2]
                    sample_df = sample_df[(sample_df['player_name'] != player)]
                    team_df.at[player, 'own'] = 1
                    lineup_pos_tracker[pos] = lineup_pos_tracker.get(pos) - 1

                    if qb_stack_players == 1:
                        player_info = sample_df[
                            (sample_df['position'] == 'WR') & (sample_df['team'] == qb_team)].sample(
                            n=1, replace=True).values
                        player = player_info[0][0]
                        player_pos = team_df.loc[player, 'position']
                        sample_df = sample_df[(sample_df['player_name'] != player)]
                        team_df.at[player, 'own'] = 1
                        if lineup_pos_tracker.get(pos) - 1 >= 0:
                            lineup_pos_tracker[player_pos] = lineup_pos_tracker.get(player_pos) - 1
                        else:
                            lineup_pos_tracker['flex'] = lineup_pos_tracker.get('flex') - 1

                    elif qb_stack_players > 1:
                        for x in range(qb_stack_players):
                            player_info = sample_df[
                                ((sample_df['position'] == 'WR') | (sample_df['position'] == 'TE') | (
                                        sample_df['position'] == 'RB')) & (sample_df['team'] == qb_team)].sample(n=1,
                                                                                                                 replace=True).values
                            player = player_info[0][0]
                            sample_df = sample_df[(sample_df['player_name'] != player)]
                            team_df.at[player, 'own'] = 1
                            player_pos = team_df.loc[player, 'position']
                            lineup_pos_tracker[player_pos] = lineup_pos_tracker.get(player_pos) - 1

                    if opp_stack_value == 1:
                        player_info = sample_df[((sample_df['position'] == 'WR') | (sample_df['position'] == 'TE') | (
                                sample_df['position'] == 'RB')) & (sample_df['team'] == qb_opp)].sample(n=1,
                                                                                                        replace=True).values
                        player = player_info[0][0]
                        sample_df = sample_df[(sample_df['player_name'] != player)]
                        team_df.at[player, 'own'] = 1
                        player_pos = team_df.loc[player, 'position']
                        lineup_pos_tracker[player_pos] = lineup_pos_tracker.get(player_pos) - 1

                elif pos == 'WR' or pos == 'RB' or pos == 'D' or pos == 'TE':
                    if remaining_salary <= 0 or remaining_players == 0:
                        break
                    if lineup_pos_tracker.get(pos) > 0:
                        if remaining_salary <= 0:
                            break
                        for x in range(lineup_pos_tracker.get(pos)):
                            try:
                                player_info = sample_df[
                                    (sample_df['position'] == pos) & (sample_df['salary'] >= avg_slary_remain_min) & (
                                            sample_df['salary'] <= avg_salary_remain_max) & (
                                                sample_df['team'] != qb_team) &
                                    (sample_df['opp'] != qb_opp)].sample(n=1, replace=True).values
                            except ValueError:
                                player_info = sample_df[(sample_df['position'] == pos)].sample(n=1, replace=True).values

                            player = player_info[0][0]
                            sample_df = sample_df[(sample_df['player_name'] != player)]
                            team_df.at[player, 'own'] = 1
                            lineup_pos_tracker[pos] = lineup_pos_tracker.get(pos) - 1
                            team_df['player_salary'] = team_df['salary'] * team_df['own']
                            total_salary = team_df['player_salary'].sum()
                            total_players = team_df['own'].sum()
                            remaining_salary = salary_cap - total_salary
                            remaining_players = total_roster - total_players
                            if remaining_salary <= 0 or remaining_players <= 0:
                                break
                            if remaining_players < 4:
                                avg_slary_remain_min = max(((remaining_salary / remaining_players) * .8),
                                                           min_viable_salary)
                                avg_salary_remain_max = (remaining_salary / remaining_players) * 1.15
                            if remaining_players == 2:
                                avg_slary_remain_min = max(remaining_salary - max_viable_salary, min_viable_salary)
                                avg_salary_remain_max = max(remaining_salary - min_viable_salary, min_viable_salary)
                            if remaining_players == 1:
                                avg_slary_remain_min = realistic_salary_floor - total_salary
                                avg_salary_remain_max = remaining_salary


                    else:
                        continue

                elif pos == 'flex':
                    if remaining_salary <= min_player_salary or remaining_players == 0:
                        break
                    salary_floor = realistic_salary_floor - total_salary
                    try:
                        player_info = sample_df[((sample_df['position'] == 'RB') | (sample_df['position'] == 'WR') | (
                                    sample_df['position'] == 'TE'))
                                                & (sample_df['salary'] <= remaining_salary) & (
                                                            sample_df['salary'] >= salary_floor) & (
                                                            sample_df['team'] != qb_team) &
                                                (sample_df['opp'] != qb_opp)].sample(n=1).values
                    except ValueError:
                        player_info = sample_df[((sample_df['position'] == 'RB') | (sample_df['position'] == 'WR') | (
                                    sample_df['position'] == 'TE'))].sample(n=1, replace=True).values
                    player = player_info[0][0]
                    sample_df = sample_df[(sample_df['player_name'] != player)]
                    team_df.at[player, 'own'] = 1
                    lineup_pos_tracker[pos] = lineup_pos_tracker.get(pos) - 1

                team_df['player_salary'] = team_df['salary'] * team_df['own']
                total_salary = team_df['player_salary'].sum()
                total_players = team_df['own'].sum()

            total_team = team_df['own'].tolist()

            total_players = team_df['own'].sum()

            if total_salary <= salary_cap and total_salary >= realistic_salary_floor and total_players == total_roster:
                total_valid_lineups = total_valid_lineups + 1
                # print(total_valid_lineups)
                final_lineups.append(total_team)
            else:
                invalid_lineup = invalid_lineup + 1
        return (final_lineups)

    def teams_mp(self):
        chunk_size = int(self.contest_size / mp.cpu_count())
        pool = mp.Pool(mp.cpu_count())
        workers = [pool.apply_async(func=self.team_creation, args=(chunk_size,)) for x in range(mp.cpu_count())]

        lineups = [p.get() for p in workers]
        lineup_list = []
        for lineup in lineups:
            lineup_list = lineup_list + lineup
        lineup_df = pd.DataFrame(lineup_list, columns=self.ownership.index.values)
        self.lineups=lineup_df

    def create_all_teams(self):
        self.initialize_variables()
        self.teams_mp()

    def player_rank(self):
        final_rank = {}
        for player in self.ownership.index.values:
            try:
                player_team = self.ownership.loc[player, 'team']
                player_pos = self.ownership.loc[player, 'pos']
                rank_df = self.ownership.loc[
                    (self.ownership['team'] == player_team) & (self.ownership['pos'] == player_pos)]
                player_team_rank = rank_df.index.get_loc(player) + 1
                if player_pos=='WR':
                    final_rank[player]=min(player_team_rank,3)
                else:
                    final_rank[player] = min(player_team_rank, 2)
            except:
                if player_pos=='WR':
                    final_rank[player]=3
                else:
                    final_rank[player] = 2

        self.player_depth=final_rank

    def correlation_matrix(self):
        correlation_matrix = []

        all_players = self.ownership.index.values


        for row_player in all_players:
            row_player_team = self.ownership.loc[row_player, 'team']
            row_player_opp = self.ownership.loc[row_player, 'opp']
            row_player_pos = self.ownership.loc[row_player, 'position']
            row_player_depth=self.player_depth.get(row_player)
            row_player_pos_depth=row_player_pos+str(row_player_depth)

            row_correlation = []
            for column_player in all_players:
                column_player_team = self.ownership.loc[column_player, 'team']
                column_player_pos = self.ownership.loc[column_player, 'position']
                column_player_depth = self.player_depth.get(column_player)
                column_player_pos_depth = column_player_pos + str(column_player_depth)

                if column_player == row_player:
                    row_correlation.append(1)
                    continue
                elif (column_player_pos=='QB' and column_player_depth>1) or (row_player_pos=='QB' and row_player_depth>1):
                    row_correlation.append(1)
                    continue
                elif column_player_team == row_player_team:
                    row_correlation.append(self.corr.loc[row_player_pos_depth, column_player_pos_depth])
                    continue

                elif column_player_team == row_player_opp:
                    column_player_pos_depth = column_player_pos_depth + 'op'
                    row_correlation.append(self.corr.loc[row_player_pos_depth, column_player_pos_depth])
                    continue
                else:
                    row_correlation.append(0)
            correlation_matrix.append(row_correlation)
        self.player_corr_matrix=correlation_matrix

    def covariance_matrix(self):
        transpose_std = [[x] for x in self.ownership['fp_std'].values]
        std_num_t = np.array(transpose_std)
        player_cov = self.player_corr_matrix * std_num_t * self.ownership['fp_std'].values
        self.player_covariance=player_cov

    def correlated_random_samples(self):
        self.player_rank()
        self.correlation_matrix()
        self.covariance_matrix()
        correlated_samples = np.random.multivariate_normal(self.ownership['fp'].values, self.player_covariance, size=self.sims)
        correlated_samples[correlated_samples < 0] = 0
        self.player_samples=correlated_samples

    def contest_sims(self):
        total_own = np.array(0)
        top_place_cutoff=max(self.top_percent*len(self.lineups),1)
        ranks=[]
        team_ids=self.lineups.index.values.tolist()*self.sims
        for sim in self.player_samples:
            self.lineups['total_fp'] = np.dot(self.lineups, sim)
            self.lineups['place'] = self.lineups['total_fp'].rank(ascending=0)
            lineup_own = np.array(self.lineups[(self.lineups['place'] <= top_place_cutoff)].values.tolist())
            total_own = total_own + (np.sum(lineup_own, axis=0))
            ranks=ranks+self.lineups['place'].values.tolist()
            del self.lineups['total_fp']
            del self.lineups['place']
        final_total_own = (total_own / (top_place_cutoff * self.sims)) * 100

        sim_ownership_list = []
        for a, b in zip(final_total_own, self.ownership.index.values.tolist()):
            sim_ownership_list.append([float(a), b])
        sim_ownership_df = pd.DataFrame(sim_ownership_list, columns=['sim_ownership', 'player_name'])
        sim_ownership_df = pd.merge(sim_ownership_df, self.ownership, on='player_name')
        lineup_created_ownership = self.lineups.mean().to_frame()
        sim_ownership_df = pd.merge(sim_ownership_df, lineup_created_ownership, left_on='player_name', right_index=True)
        sim_ownership_df.columns = ['sim_ownership', 'player_name', 'team', 'opp', 'position', 'salary', 'ownership', 'pos',
                                    'fp', 'fp_std', 'created_own']

        sim_ownership_df['created_own'] = sim_ownership_df['created_own'] * 100
        sim_ownership_df['positive_ev'] = sim_ownership_df['sim_ownership'] - sim_ownership_df['created_own']

        self.sim_ownership=sim_ownership_df
        #self.ranks=pd.DataFrame({'ids':team_ids, 'rank':ranks})

        time_start=time.time()
        self.dummy_teams()
        time_end=time.time()


        self.sim_ownership.to_csv(str(self.csv_path + "/" + "sim_ownership.csv"))
        #self.ranks.to_csv(str(self.csv_path + "/" + "ranks.csv"))
        self.opp_teams.to_csv(str(self.csv_path + "/" + "dummy_teams.csv"))

    def dummy_teams(self):
        teams=[]
        for index, row in zip(self.lineups.index.values.tolist(), self.lineups.values.tolist()):
            for row_value, column in zip(row, self.lineups.columns):
                if row_value==1:
                    teams.append([index,column, self.ownership.loc[column, 'position'], self.ownership.loc[column, 'salary']])
        dummy_teams_df=pd.DataFrame(teams, columns=['id', 'player', 'position','salary'])
        self.opp_teams=dummy_teams_df



if __name__ == '__main__':
    generator=lineup_simulator()
    generator.initialize_variables()

    time1=time.time()
    generator.teams_mp()
    time2=time.time()

    generator.correlated_random_samples()
    time3=time.time()

    generator.contest_sims()
    time4=time.time()

    print(['team time', (time2-time1)/60])
    print(['sample time', (time3 - time2) / 60])
    print(['sim time', (time4 - time3) / 60])
