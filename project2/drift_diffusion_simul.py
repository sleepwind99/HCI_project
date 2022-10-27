# Code written by June-Seop Yoon

import numpy as np

def drift_diffusion_simulation(a, mu, T_er, num_of_simul=1000):
    """
    Returns the simulation result of drift diffusion process
    for given parameter setting.

    Return reaction time and correct/error result in numpy array
    """
    eta = 0.1067
    s_z = 0.0321
    s_t = 0.0943
    sigma = 0.1
    z = a / 2
    step_size = 0.00005     # 0.05 ms

    while True:
        reaction_time = []
        correction = []

        for _ in range(num_of_simul):

            while True:
                sample_mu = np.random.normal(mu, eta)
                sample_T_er = T_er + np.random.uniform(-0.5, 0.5) * s_t
                evidence = z + np.random.uniform(-0.5, 0.5) * s_z

                if sample_mu > 0 and sample_T_er > 0 and evidence > 0 and evidence < a: break
            
            p = 0.5 * (1 + sample_mu * np.sqrt(step_size) / sigma)
            delta = sigma * np.sqrt(step_size)

            num_step = 0

            while True:
                if np.random.binomial(1, p):
                    evidence += delta
                else:
                    evidence -= delta

                num_step += 1

                if evidence >= a:
                    reaction_time.append(num_step * step_size + sample_T_er)
                    correction.append(1)
                    break
                if evidence <= 0:
                    reaction_time.append(num_step * step_size + sample_T_er)
                    correction.append(0)
                    break
        
        reaction_time, correction = np.array(reaction_time), np.array(correction)
        if np.sum(correction) > 2 and np.sum(correction) < num_of_simul - 2:
            # Each case should contain at least 3 trials.
            break

    return np.array(reaction_time), np.array(correction, dtype=int)

