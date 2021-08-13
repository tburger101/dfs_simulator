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
        self.player_depth=None
        self.player_covariance=None
        self.player_samples=None

    def initialize_variables(self):
        #Setting all variables from the config file to our lineup_simulator object
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

        ownership_loc=str(self.csv_path+"/Input Files/"+"ownership_player.csv")
        corr_loc=str(self.csv_path+"/Input Files/"+"correlation.csv")
        projection_loc = str(self.csv_path + "/Input Files/" + "player_projection_std.csv")

        ownership_df=pd.read_csv(ownership_loc)
        projections_df=pd.read_csv(projection_loc)

        #merging the ownership and projection input files
        ownership_df=pd.merge(ownership_df, projections_df, on='player_name')
        ownership_df.set_index('player_name', inplace=True)
        ownership_df.sort_values(by=['team', 'position', 'fp'], inplace=True, ascending=False)

        self.corr=pd.read_csv(corr_loc)
        self.corr.set_index('Position', inplace=True)
        self.ownership=ownership_df

    def team_creation(self, total_teams):
        '''Create the set number of dummy teams based on salary_cap, roster_const, total_roster, stack_player_percent,  and
        opp_stack_percent from the config file to compete against each other'''

        total_valid_lineups, invalid_lineup = 0, 0
        ownership_df=self.ownership

        lineup_construction, salary_cap, total_roster, stack_player_percent,opp_stack_percent=self.roster_construction, self.salary_cap, \
                self.total_roster, self.stack_player_percent, self.opp_stack_percent
        ownership_list = []
        final_lineups = []

        #creating a list of players which will eventually become a dataframe we can sample from to create teams
        #AKA total_probability_sample

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
                    #Determining how many players to stack with from the QB team based on config setting probability
                    stack_values = np.random.multinomial(1, stack_player_percent)

                    # Determining how many players to stack with from the QB opp team based on config setting probability
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

                    #In future can apply stricter rules on position types  to stack
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

                #Right now the only rule is don't stack an opposing defense against a players qb. Can
                #add more rules in the future to make the team construction smarter like with an ineligible team variable

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
                                            sample_df['salary'] <= avg_salary_remain_max) & ~(
                                                sample_df['team'].isin([qb_team, qb_opp]))].sample(n=1, replace=True).values
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
                                                            sample_df['salary'] >= salary_floor) & ~(
                                                            sample_df['team'].isin([qb_opp,qb_team]))].sample(n=1).values
                    except ValueError:
                        player_info = sample_df[((sample_df['position'] == 'RB') | (sample_df['position'] == 'WR') | (
                                    sample_df['position'] == 'TE'))& ~(sample_df['team'].isin([qb_opp,qb_team]))].sample(n=1).values
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
        #Create a pool of workers equal to CPU count. These will be used to create dummy teams
        chunk_sizes = self.split_sims(self.contest_size)
        chunk_sizes=[x[1]-x[0] for x in chunk_sizes]
        pool = mp.Pool(mp.cpu_count())
        workers = [pool.apply_async(func=self.team_creation, args=(x,)) for x in chunk_sizes]

        lineups = [p.get() for p in workers]
        lineup_list = []
        for lineup in lineups:
            lineup_list = lineup_list + lineup
        lineup_df = pd.DataFrame(lineup_list, columns=self.ownership.index.values)
        self.lineups=lineup_df

    def player_rank(self):
        '''Determine player depth (WR1, WR2, etc) to be used when figuring out our correlation matrix. The ownership variable has
        already been sorted by team, position, and fantasy points which makes it easy to determine pos rank for
        each player'''

        final_rank = {}
        #Max depth for WR is 3. Max depth for all other positions are set to 2.
        # QB depth of 2 is handled in the correlation matrx and treated as 0 correlation with every position
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
        '''This is where the correlation matrix given with our input files can be combined with our determined
        player  ranks and create a player specific correlation matrix'''

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

                #All back up QBs are assumed to have 0 correlation with every player
                elif (column_player_pos=='QB' and column_player_depth>1) or (row_player_pos=='QB' and row_player_depth>1):
                    row_correlation.append(0)
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
        return(correlation_matrix)

    def covariance_matrix(self, ):
        '''Combining the player correlation and std dev estimates from the input file to create a covariance matrix to be used
        for monte carlo simulations'''

        player_corr_matrix=self.correlation_matrix()
        transpose_std = [[x] for x in self.ownership['fp_std'].values]
        std_num_t = np.array(transpose_std)
        player_cov = player_corr_matrix * std_num_t * self.ownership['fp_std'].values
        self.player_covariance=player_cov

    def correlated_random_samples(self):
        #Creating the correlated
        self.player_rank()
        self.correlation_matrix()
        self.covariance_matrix()
        correlated_samples = np.random.multivariate_normal(self.ownership['fp'].values, self.player_covariance, size=self.sims)

        #Since we have assumed player points follow a normal distribution (they don't except for maybe QB) in order to
        #get rid of the large negative numbers sample fantasty points are bounded at 0

        correlated_samples[correlated_samples < 0] = 0
        self.player_samples=correlated_samples

    def contest_sims(self):
        '''Simulating the contest using our player samples and determining what place everyone finished in'''

        #In order to increase performance the simulations utilize a pool of worksers
        intervals=self.split_sims(self.sims)

        pool = mp.Pool(mp.cpu_count())
        workers = [pool.apply_async(func=self.sim_rankings, args=(sim_range,)) for sim_range in intervals]
        pool.close()
        sim_results_data = [p.get() for p in workers]
        sim_results=[x.get('total_own') for x in sim_results_data]
        sim_teams=[x.get('total_teams') for x in sim_results_data]

        total_own=np.sum(sim_results, axis=0)
        final_total_own = (total_own / np.sum(np.array(sim_teams))) * 100

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

        self.dummy_teams()
        self.create_rankings(sim_results_data)

        sim_ownership_df.to_csv(str(self.csv_path + "/Output Files/" + "sim_ownership.csv"), index=False)

    def sim_rankings(self, sim_range):
        '''
        When the contest is simulated we are collecting data on three things.
            1. What place did all the teams finish in and convert that to a bin or percentile.
            For example bin 0 corresponds to top 1% AKA total_ranks
            2. How many total lineups are inside the top percent set in the config file.
            This seems easy but needs to be calculated because there can be ties decimal top place cut off
            based on the size of the contest AKA total_teams
            3. Ownership of each player in the top percent AKA total_own
            '''


        ranks=[]
        total_own = np.array(0)
        total_teams=0
        top_place_cutoff = max(self.top_percent * len(self.lineups), 1)
        total_sims=sim_range[1]-sim_range[0]
        total_bins=int(1/self.top_percent)

        for x in range(sim_range[0], sim_range[1]):
            sim=self.player_samples[x]
            self.lineups['total_fp'] = np.dot(self.lineups, sim)
            self.lineups['place'] = self.lineups['total_fp'].rank(ascending=0)
            lineup_own = np.array(self.lineups[(self.lineups['place'] <= top_place_cutoff)].values.tolist())
            total_teams=len(self.lineups[(self.lineups['place'] <= top_place_cutoff)])+ total_teams
            total_own = total_own + (np.sum(lineup_own, axis=0))
            ranks=ranks+ (pd.cut(self.lineups['place'], bins=total_bins, labels=False)).tolist()
            del self.lineups['total_fp']
            del self.lineups['place']

        rank_df=pd.DataFrame({'team_id': self.lineups.index.values.tolist()*total_sims, 'rank_bin':ranks})
        total_ranks=rank_df.pivot_table(index='team_id', columns='rank_bin', aggfunc={'rank_bin':len}, fill_value=0).values
        return({'total_own':total_own, 'total_teams':total_teams, 'ranks':total_ranks})

    def split_sims(self, total_items):
        '''Determining based on the number of CPU cores and the total number of items to split how to split
        these for our multiprocessing pool'''

        cores=mp.cpu_count()
        split_interval=int(total_items/cores)
        beg_interval=0
        intervals=[]
        for x in range(0,cores):
            ending_interval=beg_interval+split_interval
            intervals.append([beg_interval, ending_interval])
            beg_interval=ending_interval
        intervals[cores-1][1]=total_items
        return(intervals)

    def create_rankings(self, sim_reults_data):
        #Create a rankings df based on percentile/bin in the output file
        total_rank = np.array(0)
        for rank in sim_reults_data:
            total_rank=total_rank+rank.get('ranks')
        final_rank_df=pd.DataFrame(total_rank)

        final_rank_df.columns=['bin'+str(x) for x in final_rank_df.columns]
        final_rank_df.to_csv(str(self.csv_path + "/Output Files/" + "ranks.csv"), index=True)

    def dummy_teams(self):
        #Output the dummy teams to a CSV
        teams=[]
        for index, row in zip(self.lineups.index.values.tolist(), self.lineups.values.tolist()):
            for row_value, column in zip(row, self.lineups.columns):
                if row_value==1:
                    teams.append([index,column, self.ownership.loc[column, 'position'], self.ownership.loc[column, 'salary']])
        dummy_teams_df=pd.DataFrame(teams, columns=['id', 'player', 'position','salary'])
        dummy_teams_df.to_csv(str(self.csv_path + "/Output Files/" + "dummy_teams.csv"), index=False)

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
