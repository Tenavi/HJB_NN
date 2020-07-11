##### Change this line for different a different problem. #####

system = 'satellite'
#system = 'burgers'

time_dependent = False #True

if system == 'satellite':
    from examples.satellite.problem_def import setup_problem, config_NN
elif system == 'burgers':
    from examples.burgers.problem_def import setup_problem, config_NN

problem = setup_problem()
config = config_NN(problem.N_states, problem.t1, time_dependent)

if system == 'burgers':
    system += '/D' + str(problem.N_states)
