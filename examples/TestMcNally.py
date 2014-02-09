#!/usr/bin/python
from __future__ import division

import argparse
import json
import os
import sys
import time
import random as rnd
import commands as comm

import numpy as np
import cPickle as pickle
import MultiNEAT as NEAT
import multiprocessing as mpc

import scipy
from scipy.stats import nbinom

import h5py


# This isn't necessarily the default values,
# just need something to initialize these
# variables
DEFECTOR_PAYOFF           = 7 # T
MUTUAL_COOPERATION_PAYOFF = 6 # R
MUTUAL_DEFECTION_PAYOFF   = 2 # P
COOPERATOR_PAYOFF         = 1 # S

DEFECT = 0
COOPERATE = 1

PAYOFFS = [[MUTUAL_DEFECTION_PAYOFF, DEFECTOR_PAYOFF],
           [COOPERATOR_PAYOFF, MUTUAL_COOPERATION_PAYOFF]]

INTEL_PENALTY = 0.01

# These classes aren't used yet..
class CoopDefectGame(object):
    def __init__(self):
        DEFECTOR_PAYOFF           = 0.0 # T
        MUTUAL_COOPERATION_PAYOFF = 0.0 # R
        MUTUAL_DEFECTION_PAYOFF   = 0.0 # P
        COOPERATOR_PAYOFF         = 0.0 # S

        self._setup_payoff_table()

    def _setup_payoff_table(self):
        self.payoffs = \
          [[self.mutual_defection, self.defector],
           [self.cooperator, self.mutual_cooperation]]

    def set_defector_payoff(self, T):
        self.defector_payoff = T

    def set_mutual_cooperation_payoff(self, R):
        self.mutual_cooperation_payoff = R

    def set_mutual_defection_payoff(self, P):
        self.defector_payoff = P

    def get_payoff(self, action, opp_action):
        return self.payoffs[action][opp_action]

class IPDGame(CoopDefectGame):
    def __init__(self):
        # T > R > P > S
        self.defector           = 7.0 # T
        self.mutual_cooperation = 6.0 # R
        self.mutual_defection   = 2.0 # P
        self.cooperator         = 2.0 # S

        self._setup_payoff_table()

class ISDGame(CoopDefectGame):
    def __init__(self):
        # T > R > S > P
        self.defector           = 8.0 # T
        self.mutual_cooperation = 5.0 # R
        self.mutual_defection   = 1.0 # P
        self.cooperator         = 2.0 # S

        self._setup_payoff_table()

# Agent class that represents a player in a
# coop/defect game. 
class Agent(object):
    def __init__(self):
        self.total_payoff = 0

    def get_action(self, prev_payoffs):
        pass

    def get_total_payoff(self):
        return self.total_payoff

    def add_payoff(self, payoff):
        self.total_payoff += payoff

    def flush(self):
        pass

class AlwaysCooperateAgent(Agent):
    def get_action(self, prev_payoffs):
        return COOPERATE

class AlwaysDefectAgent(Agent):
    def get_action(self, prev_payoffs):
        return DEFECT

class TitForTatAgent(Agent):
    def get_action(self, prev_payoffs):
        # Default cooperate, but defect if opponent deffected
        # in the previous round.
        if prev_payoffs[1] == DEFECTOR_PAYOFF or \
           prev_payoffs[1] == MUTUAL_DEFECTION_PAYOFF:
            return DEFECT
        else:
            return COOPERATE

class TitForTwoTatsAgent(Agent):
    def __init__(self):
        self.opponent_defected = False

    def flush(self):
        self.opponent_defected = False

    def get_action(self, prev_payoffs):
        # Default cooperate, but defect if opponent deffected
        # in the previous two rounds. Member variable
        # `opponent_defected` records whether or not the
        # opponent defected two rounds back.
        if prev_payoffs[1] == DEFECTOR_PAYOFF or \
           prev_payoffs[1] == MUTUAL_DEFECTION_PAYOFF:
            if self.opponent_defected:
                return DEFECT
            else:
                self.opponent_defected = True
                return COOPERATE
        else:
            self.opponent_defected = False
            return COOPERATE

class PavlovAgent(Agent):
    def get_action(self, prev_payoffs):
        # Default cooperate, but defect if opponent deffected
        # in the previous two rounds. Member variable
        # `opponent_defected` records whether or not the
        # opponent defected two rounds back.
        if prev_payoffs[0] == DEFECTOR_PAYOFF or \
           prev_payoffs[0] == COOPERATOR_PAYOFF:
            return DEFECT
        else:
            return COOPERATE

class ProbabilisticAgent(Agent):
    def __init__(self, prob):
        self.prob = prob

    def get_action(self, prev_payoffs):
        # Cooperate with probability `self.prob`
        r = np.random.rand()
        if r <= self.prob:
            return COOPERATE
        else:
            return DEFECT

class NeuralNetworkAgent(Agent):
    def __init__(self, net=None):
        self.net = net

        self.total_payoff = 0

    # prev_payoffs[0] should be the agent's previous payoff.
    # prev_payoffs[1] should be the opponents's previous payoff.
    def get_action(self, prev_payoffs):
        inputs = list(prev_payoffs)
        # Don't forget to include the bias input (always 1)
        inputs.append(1.0)
        self.net.Input(inputs)
        self.net.Activate()
        outputs = self.net.Output()

        r = np.random.rand()
        if r < outputs[0]:
            return COOPERATE
        else:
            return DEFECT

    def get_total_payoff(self):
        return self.total_payoff

    def add_payoff(self, payoff):
        self.total_payoff += payoff

    def flush(self):
        self.net.Flush()


# Tests an agent to determine which of several
# standard strategies it is most similar to.
def test_agent(agent):
    opp_moves = []
    agent_moves = []
    ac_moves = []
    ad_moves = []
    tft_moves = []
    tftt_moves = []
    pavlov_moves = []

    probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    for p in probs:
        for i in range(5):
            agent.flush()
            opp = ProbabilisticAgent(p)
            tft_agent = TitForTatAgent()
            tftt_agent = TitForTwoTatsAgent()
            pavlov_agent = PavlovAgent()
            agent_payoff = 0
            opp_payoff = 0
            tft_payoff = 0
            tftt_payoff = 0
            pavlov_payoff = 0
            for j in range(20):
                agent_decision = agent.get_action([agent_payoff, opp_payoff])
                opp_decision = opp.get_action([opp_payoff, agent_payoff])
                tft_decision = tft_agent.get_action([tft_payoff, opp_payoff])
                tftt_decision = tftt_agent.get_action([tftt_payoff, opp_payoff])
                pavlov_decision = pavlov_agent.get_action([pavlov_payoff, opp_payoff])

                opp_moves.append(opp_decision)
                agent_moves.append(agent_decision) 
                ac_moves.append(COOPERATE)
                ad_moves.append(DEFECT)
                tft_moves.append(tft_decision)
                tftt_moves.append(tftt_decision)
                pavlov_moves.append(pavlov_decision)

                agent_payoff = PAYOFFS[agent_decision][opp_decision]
                opp_payoff = PAYOFFS[opp_decision][agent_decision]
                tft_payoff = PAYOFFS[tft_decision][opp_decision]
                tftt_payoff = PAYOFFS[tftt_decision][opp_decision]
                pavlov_payoff = PAYOFFS[pavlov_decision][opp_decision]

    # Find closest standard strategy to the agent's strategy
    agent_moves = np.array(agent_moves)
    ac_moves = np.array(ac_moves)
    ad_moves = np.array(ad_moves)
    tft_moves = np.array(tft_moves)
    tftt_moves = np.array(tftt_moves)
    pavlov_moves = np.array(pavlov_moves)

    ac_dist     = np.sum(np.abs(agent_moves - ac_moves))
    ad_dist     = np.sum(np.abs(agent_moves - ad_moves))
    tft_dist    = np.sum(np.abs(agent_moves - tft_moves))
    tftt_dist   = np.sum(np.abs(agent_moves - tftt_moves))
    pavlov_dist = np.sum(np.abs(agent_moves - pavlov_moves))
    dists = np.array([ac_dist, ad_dist, tft_dist, tftt_dist, pavlov_dist])

    return np.argmin(dists), np.sum(agent_moves) / float(len(agent_moves))

def evaluate_iterated_game(genomes):
    # When using this evaluation function, a list of lists of genomes
    # should be provided to the main 'evaluate' function.
    agents = []
    for g in genomes:
        net = NEAT.NeuralNetwork()
        g.BuildPhenotype(net)
        agents.append(NeuralNetworkAgent(net))

    fitness = 0
    n_total_rounds = 0.0
    p1 = agents[0]
    for p2 in agents[1:]:
        n_rounds = nbinom.rvs(1, 0.02, 1) + 1
        p1.flush()
        p2.flush()
        p1_payoff = 0
        p2_payoff = 0
        for i in range(n_rounds):
            p1_decision = p1.get_action([p1_payoff, p2_payoff]);
            p2_decision = p2.get_action([p2_payoff, p1_payoff]);

            p1_payoff = PAYOFFS[p1_decision][p2_decision]
            p2_payoff = PAYOFFS[p2_decision][p1_decision]

            p1.add_payoff(p1_payoff)
            n_total_rounds += 1.0

    fitness = p1.get_total_payoff() / n_total_rounds
    fitness -= INTEL_PENALTY*(len(p1.net.neurons))
    return fitness

def run_experiment(n_generations, params, save_freq, save_file=None):
    # NEAT.Genome(uint a_ID, uint a_NumInputs, uint a_NumHidden,
    #             uint a_NumOutputs, bool a_FS_NEAT,
    #             ActivationFunction a_OutputActType,
    #             ActivationFunction a_HiddenActType,
    #             uint a_SeedType, // 1 -> use a_NumHidden, 0 -> ignore a_NumHidden
    #             Parameters a_Parameters);
    g = NEAT.Genome(0, 3, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 1, params)
    pop = NEAT.Population(g, params, True, 1.0)
    
    pool = mpc.Pool(processes = 8)

    sel_for_intel = np.zeros((n_generations, 1))
    corr_intel_coop = np.zeros((n_generations, 1))
    corr_intel_fit = np.zeros((n_generations, 1))
    corr_coop_fit = np.zeros((n_generations, 1))

    fit_max = np.zeros((n_generations, 1))
    fit_min = np.zeros((n_generations, 1))
    fit_mean= np.zeros((n_generations, 1))
    fit_std = np.zeros((n_generations, 1))

    intel_max = np.zeros((n_generations, 1))
    intel_min = np.zeros((n_generations, 1))
    intel_mean= np.zeros((n_generations, 1))
    intel_std = np.zeros((n_generations, 1))

    coop_max = np.zeros((n_generations, 1))
    coop_min = np.zeros((n_generations, 1))
    coop_mean= np.zeros((n_generations, 1))
    coop_std = np.zeros((n_generations, 1))

    strat_ac = np.zeros((n_generations, 1))
    strat_ad = np.zeros((n_generations, 1))
    strat_tft = np.zeros((n_generations, 1))
    strat_tftt = np.zeros((n_generations, 1))
    strat_pavlov = np.zeros((n_generations, 1))

    if not save_file is None:
        sel_for_intel_dset = save_file.create_dataset('sel_for_intel', (n_generations, 1))

        corr_intel_coop_dset = save_file.create_dataset('corr/intel_coop', (n_generations, 1))
        corr_intel_fit_dset = save_file.create_dataset('corr/intel_fit', (n_generations, 1))
        corr_coop_fit_dset = save_file.create_dataset('corr/coop_fit', (n_generations, 1))

        fit_max_dset = save_file.create_dataset('fit/max', (n_generations, 1))
        fit_min_dset = save_file.create_dataset('fit/min', (n_generations, 1))
        fit_mean_dset= save_file.create_dataset('fit/mean', (n_generations, 1))
        fit_std_dset = save_file.create_dataset('fit/std', (n_generations, 1))

        intel_max_dset = save_file.create_dataset('intel/max', (n_generations, 1))
        intel_min_dset = save_file.create_dataset('intel/min', (n_generations, 1))
        intel_mean_dset= save_file.create_dataset('intel/mean', (n_generations, 1))
        intel_std_dset = save_file.create_dataset('intel/std', (n_generations, 1))

        coop_max_dset = save_file.create_dataset('coop/max', (n_generations, 1))
        coop_min_dset = save_file.create_dataset('coop/min', (n_generations, 1))
        coop_mean_dset= save_file.create_dataset('coop/mean', (n_generations, 1))
        coop_std_dset = save_file.create_dataset('coop/std', (n_generations, 1))

        strat_ac_dset = save_file.create_dataset('strategy/ac', (n_generations, 1))
        strat_ad_dset = save_file.create_dataset('strategy/ad', (n_generations, 1))
        strat_tft_dset = save_file.create_dataset('strategy/tft', (n_generations, 1))
        strat_tftt_dset = save_file.create_dataset('strategy/tftt', (n_generations, 1))
        strat_pavlov_dset = save_file.create_dataset('strategy/pavlov', (n_generations, 1))

        n_completed_dset = save_file.create_dataset('n_generations', (1,))
        n_completed_dset[0] = n_generations

    for generation in range(n_generations):
        genome_list = NEAT.GetGenomeList(pop)

        genome_lol = []
        for i in range(len(genome_list)):
            new_list = [genome_list[i]]
            new_list.extend(genome_list[0:i])
            new_list.extend(genome_list[i+1:])
            genome_lol.append(new_list)

        fitness_list = NEAT.EvaluateGenomeList_Parallel(genome_lol, evaluate_iterated_game)
        #fitness_list = NEAT.EvaluateGenomeList_Serial(genome_lol, evaluate_iterated_game)
        NEAT.ZipFitness(genome_list, fitness_list)
        
        best = max([x.GetLeader().GetFitness() for x in pop.Species])
        print 'Generation: ', generation
        print ' Best fitness:', best#, 'Species:', len(pop.Species)
        print ' # Species:', len(pop.Species)
        print ' # Individuals:', sum([len(s.Individuals) for s in pop.Species])
        
        # test
        net = NEAT.NeuralNetwork()
        #pop.Species[0].GetLeader().BuildPhenotype(net)
        distribution = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        intelligences = []
        cooperation_freq = []
        fitnesses = []
        for s in pop.Species:
            for i in s.Individuals:
                i.BuildPhenotype(net)
                intelligences.append(len(net.neurons))
                (winner, coop) = test_agent(NeuralNetworkAgent(net))
                distribution[winner] += 1
                cooperation_freq.append(coop)
        distribution /= np.sum(distribution)
        intelligences = np.array(intelligences, dtype=np.float)
        intelligence = np.mean(intelligences)

        sel_for_intel[generation] = np.cov(intelligences, fitness_list)[0,1] / np.mean(fitness_list)
        corr_intel_coop[generation] = np.corrcoef(intelligences, cooperation_freq)[0, 1]
        corr_intel_fit[generation] = np.corrcoef(intelligences, fitness_list)[0, 1]
        corr_coop_fit[generation] = np.corrcoef(cooperation_freq, fitness_list)[0, 1]

        fit_max[generation] = np.max(fitness_list)
        fit_min[generation] = np.min(fitness_list)
        fit_mean[generation] = np.mean(fitness_list)
        fit_std[generation] = np.std(fitness_list)

        intel_max[generation] = np.max(intelligences)
        intel_min[generation] = np.min(intelligences)
        intel_mean[generation] = np.mean(intelligences)
        intel_std[generation] = np.std(intelligences)

        coop_max[generation] = np.max(cooperation_freq)
        coop_min[generation] = np.min(cooperation_freq)
        coop_mean[generation] = np.mean(cooperation_freq)
        coop_std[generation] = np.std(cooperation_freq)

        strat_ac[generation] = distribution[0]
        strat_ad[generation] = distribution[1]
        strat_tft[generation] = distribution[2]
        strat_tftt[generation] = distribution[3]
        strat_pavlov[generation] = distribution[4]

        print ' Selection for intelligence:  ', sel_for_intel[generation]
        print ' Correlation of Int and Coop: ', corr_intel_coop[generation]
        print ' Strategy distribution: ', distribution
        print ' Fitness:      Max={:.2f} Min={:.2f} Avg={:.2f} Std={:.2f}'.format(
                np.max(fitness_list),
                np.min(fitness_list),
                np.mean(fitness_list),
                np.std(fitness_list))
        print ' Intelligence: Max={:.2f} Min={:.2f} Avg={:.2f} Std={:.2f}'.format(
                np.max(intelligences),
                np.min(intelligences),
                np.mean(intelligences),
                np.std(intelligences))
        print ' Cooperation:  Max={:.2f} Min={:.2f} Avg={:.2f} Std={:.2f}'.format(
                np.max(cooperation_freq),
                np.min(cooperation_freq),
                np.mean(cooperation_freq),
                np.std(cooperation_freq))
        print ''

        if not save_file is None and not (generation + 1) % save_freq:
            sel_for_intel_dset[...] = sel_for_intel

            corr_intel_coop_dset[...] = corr_intel_coop
            corr_intel_fit_dset[...] = corr_intel_fit
            corr_coop_fit_dset[...] = corr_coop_fit

            fit_max_dset[...] = fit_max
            fit_min_dset[...] = fit_min
            fit_mean_dset[...] = fit_mean
            fit_std_dset[...] = fit_std

            intel_max_dset[...] = intel_max
            intel_min_dset[...] = intel_min
            intel_mean_dset[...] = intel_mean
            intel_std_dset[...] = intel_std

            coop_max_dset[...] = coop_max
            coop_min_dset[...] = coop_min
            coop_mean_dset[...] = coop_mean
            coop_std_dset[...] = coop_std

            strat_ac_dset[...] = strat_ac
            strat_ad_dset[...] = strat_ad
            strat_tft_dset[...] = strat_tft
            strat_tftt_dset[...] = strat_tftt
            strat_pavlov_dset[...] = strat_pavlov

            n_completed_dset[0] = generation + 1

            save_file.flush()
            
        pop.Epoch()

    # Create h5py datasets
    if not save_file is None:
        sel_for_intel_dset[...] = sel_for_intel

        corr_intel_coop_dset[...] = corr_intel_coop
        corr_intel_fit_dset[...] = corr_intel_fit
        corr_coop_fit_dset[...] = corr_coop_fit

        fit_max_dset[...] = fit_max
        fit_min_dset[...] = fit_min
        fit_mean_dset[...] = fit_mean
        fit_std_dset[...] = fit_std

        intel_max_dset[...] = intel_max
        intel_min_dset[...] = intel_min
        intel_mean_dset[...] = intel_mean
        intel_std_dset[...] = intel_std

        coop_max_dset[...] = coop_max
        coop_min_dset[...] = coop_min
        coop_mean_dset[...] = coop_mean
        coop_std_dset[...] = coop_std

        strat_ac_dset[...] = strat_ac
        strat_ad_dset[...] = strat_ad
        strat_tft_dset[...] = strat_tft
        strat_tftt_dset[...] = strat_tftt
        strat_pavlov_dset[...] = strat_pavlov

        n_completed_dset[0] = n_generations

        save_file.close()

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generations',
                        help='Number of generations to run the simulation',
                        dest='n_generations',
                        type=int,
                        required=True,
                        action='store')
    parser.add_argument('--save-freq',
                        help='How many generations between saving partial results',
                        dest='save_freq',
                        type=int,
                        default=10,
                        required=False,
                        action='store')
    parser.add_argument('-f', '--save-fname',
                        help='Filename to save results to (HDF5 file)',
                        dest='save_fname',
                        required=True,
                        action='store')
    parser.add_argument('-p', '--params-fname',
                        help='Filename to load simulation params from',
                        dest='params_fname',
                        required=False,
                        action='store')
    parser.add_argument('-i', '--intel-penalty',
                        help='Coefficient of intelligence (# neurons) in fitness calculation',
                        dest='intel_penalty',
                        type=float,
                        default=0.01,
                        required=False,
                        action='store')

    game_type_grp = parser.add_mutually_exclusive_group(required=True)
    game_type_grp.add_argument('--ipd',
                               help='Have individuals play the iterated prisoner\'s dilemma',
                               dest='ipd',
                               action='store_true')
    game_type_grp.add_argument('--isd',
                               help='Have individuals play the iterated snowdrift game',
                               dest='isd',
                               action='store_true')
    game_type_grp.add_argument('--custom',
                               help='Have individuals play a game with custom payoffs\n\
                                     (order: T (D), R (MC), P (MD), S (C))',
                               dest='custom_payoffs',
                               metavar=('T', 'R', 'P', 'S'),
                               type=int,
                               nargs=4,
                               action='store')
    args = parser.parse_args()

    #save_file = h5py.File('test.h5', 'w')
    save_file = h5py.File(args.save_fname, 'w')

    params = NEAT.Parameters()
    if args.params_fname is not None:
        ret_val = params.Load(args.params_fname)
        if ret_val < 0:
            print 'Parameter file not found! Exiting...'
            exit()

    if args.intel_penalty is not None:
        INTEL_PENALTY = args.intel_penalty

    if args.ipd:
        # T > R > P > S
        DEFECTOR_PAYOFF           = 7 # T
        MUTUAL_COOPERATION_PAYOFF = 6 # R
        MUTUAL_DEFECTION_PAYOFF   = 2 # P
        COOPERATOR_PAYOFF         = 1 # S
    elif args.isd:
        # T > R > S > P
        DEFECTOR_PAYOFF           = 8 # T
        MUTUAL_COOPERATION_PAYOFF = 5 # R
        MUTUAL_DEFECTION_PAYOFF   = 1 # P
        COOPERATOR_PAYOFF         = 2 # S
    else:
        # Custom payoffs
        DEFECTOR_PAYOFF           = args.custom_payoffs[0] # T
        MUTUAL_COOPERATION_PAYOFF = args.custom_payoffs[1] # R
        MUTUAL_DEFECTION_PAYOFF   = args.custom_payoffs[2] # P
        COOPERATOR_PAYOFF         = args.custom_payoffs[3] # S

    PAYOFFS = [[MUTUAL_DEFECTION_PAYOFF, DEFECTOR_PAYOFF],
               [COOPERATOR_PAYOFF, MUTUAL_COOPERATION_PAYOFF]]

    rng = NEAT.RNG()
    rng.TimeSeed()

    # Run experiment.
    run_experiment(args.n_generations, params, args.save_freq, save_file=save_file)
