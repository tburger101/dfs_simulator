# dfs_simulator
This python code is designed to simulate an NFL DFS contest. The process is outlined below.

1. Based on the paramerters set in the config.txt file dummy teams are created using qb stacking constraints and random sampling percentages in the ownership_player.csv file
2. Determine position depth of all players
3. After the teams are created and position depth is determined correlated random samples are drawn based on the correlation.csv and player_projection_std files
4. For each simulation the lineups are ranked and determined which end up in the top x percent set in the config.txt file

## Required Libraries
* numpy
* pandas

## Data Files
### config.txt file
- salary_cap- total salary cap for the dummy teams
- roster_const- list of tupples which detail the position followed by how many of that position the roster need to include
- total_roster- total roster positions
- stack_player_percent- list of decimal percentages which show how often a qb is stacked with his teammates. 1st number represents no stack, 2nd single stack, 3rd double stack, and 4th tripple stack. Must sum to 1 
- opp_stack_percent- How often the QB is stacked with an opponent
- contest_size- How many dummy teams you want to create and simulate against each other
- num_sims- Number of monte carlo simulations to run.
- top_percent- The top percent of teams you want to look at
- path- Where you store the dfs_contest_sims folder on your local machine 

### Input Files - Configurable by user
#### correlation.csv- Shows how each position is correlated
#### ownership_player.csv - Projected ownership of contest
- player_name- Name of player
- team - Player team
- opp - Player opponent
- position- Player position
- salary - Player salary
- ownership - projected ownership of the contest
#### player_projection_std.csv- Projected player fantasy points and standard deviations
- player_name- Name of player
- pos - Position of player
- fp - Projected fantasy points
- fp_std - Estimated player standard deviation

### Output Files - Files the sytem generates to summarize the result of the sims
#### dummy_teams.csv
- id - Team ID reference created by the simulation engine
- player - Name of player
- position - Position of player
- salary - Salary of player

#### sim_ownership.csv
- sim_ownership - The percentage of top lineups the player was in. Top lineup percent is in the config.txt file
- player_name - Name of player
- team - Player  team
- opp - Player opponent
- position - Player position
- salary - Player salary
- ownership - Projected ownership for the contest. Number is from the ownership_player.csv
- pos - Player position
- fp - Player projected fantasy points. Number is from the player_projection_std.csv
- fp_std - Player projected standard deviation. Number is from the player_projection_std.csv
- created_own - Total percentage of lineups the player is in based on our dummy teams
- positive_ev - sim_ownership minus created_own. Shows the how the absolute different the player outperformed his created ownership

#### ranks.csv
- team_id- Id which corresponds to a particular dummy team. Will map to the dummy_teams.csv id.
- top_percent - Percent of time the team finished in the user top_percent specified in the config.txt file.

## Running the simulations
- First edit all your input files based on your own estimations
- Run the file sim.py

