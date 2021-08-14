import contest_simlulator as sm


if __name__ == '__main__':
    generator=sm.lineup_simulator()
    generator.initialize_variables()

    generator.teams_mp()

    generator.correlated_random_samples()
    generator.contest_sims()

