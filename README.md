# dfs_simulator
This python code is designed to simulate an NFL DFS contest. The process is outline below.

1.) Based on the paramerters set in the config.txt file dummy teams are created using qb stacking constraints and random sampling percentages in the ownership_player.csv file
2.) Determine position depth of all players
3.) After the teams are created and position depth is determined correlated random samples are drawn based on the correlation.csv and player_projection_std files
4.) For each simulation the lineups are ranked and determined which end up in the top x percent set in the config.txt file

## Required Libraries
* numpy
* pandas

## config.txt files


## Input Files
* 
