import OptimalNetwork
import numpy as np


# Size of the domain (grid size)
domain_size = (50, 50)

# The exponent m
m = 0.5

# Domain geometry: 4sided or 2sided
domain_type = '4sided'

if __name__ == "__main__":
    OCN = OptimalNetwork.OCN(domain_size=domain_size,
                             m_exponent=m,
                             domain_type=domain_type,
                             initial_max_elevation=0.1,
                             initial_std=0.001)
    OCN.initiate()
    OCN.local_network()
    OCN.optimize(max_itr_no_imp=500)
    OCN.plot_results()


