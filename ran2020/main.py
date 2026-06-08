import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from mpd import *  
from tenpy.models.spins import SpinChain
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS as TenPyMPS

# ===================== System parameters =====================
N = 24                  # site number（N=24~150）
J = 1.0                 
d = 2                   # physical dimension
chi_max = 256           # max virtual dimension
D_max = 8               # max circuit layer（D=9）

# ===================== TeNPy DMRG parameters =====================
DMRG_chi = 64           # DMRG max virtual dimension
DMRG_sweeps = 10        # DMRG sweep
DMRG_max_E_err = 1e-10  
DMRG_max_S_err = 1e-6

# def tenpy_mps_to_standard_mps(tenpy_mps):
#     """
#     Turn the TeNPy MPS to MPS form in mpd code (left_virtual, physical, right_virtual)
#     """
#     standard_mps = []
#     N = tenpy_mps.L
    
#     for i in range(N):
#         tensor = tenpy_mps.get_B(i, form='A')  
#         standard_mps.append(tensor.to_ndarray())
    
#     return standard_mps

def get_heisenberg_ground_state_tenpy(N, J, chi_max, verbose=True):
    """
    calculate one dimension Heisenberg model ground state using TeNPy
    """
    if verbose:
        print("===== Calculate Heisenberg Model Ground State Using TeNPy =====")
        print(f"System size: N={N}, Coupling constant: J={J}")
        print(f"DMRG max virtual dimension: χ={chi_max}")
    
    # define Heisenberg model
    model_params = {
        'L': N,
        'S': 0.5, 
        'Jx': J,
        'Jy': J,
        'Jz': J,
        'hz': 0,
        'bc_MPS': 'finite',  
        'conserve': None     
    }
    model = SpinChain(model_params)
    
    # randomly initialize MPS
    psi = TenPyMPS.from_product_state(model.lat.mps_sites(), ['up'] * N, bc='finite')
    
    # set DMRG parameters
    dmrg_params = {
        'mixer': True,
        'max_sweeps': DMRG_sweeps,
        'max_E_err': DMRG_max_E_err,
        'max_S_err': DMRG_max_S_err,
        'chi_list': {0: 16, 5: 32, 8: chi_max},  
        'verbose': verbose
    }
    
    # run DMRG
    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = eng.run()
    
    if verbose:
        print(f"\nDMRG convergence finished！")
        print(f"Ground state total energy: {E:.10f}")
        print(f"Energy of each site: {E/N:.10f}")
        print(f"Max entropy of entanglement: {np.max(psi.entanglement_entropy()):.6f}")
    
    standard_mps = psi
    
    return standard_mps, E


if __name__ == '__main__':
    # 1. calculate ground state 
    ground_state_mps, E0 = get_heisenberg_ground_state_tenpy(N, J, DMRG_chi)
    
    # 2. execute MPS encoding algorithm
    gates, NLF = encoding_algorithm(ground_state_mps, D_max, d, N, chi_max)
    
    # 3. calculate global entanglement as comparison
    psi_trivial = canonical_compress_mps(ground_state_mps, max_chi=1)
    fid0 = get_all_zero_amplitude(psi_trivial)**2
    F0 = -np.log(fid0 + 1e-30) / N
    print(f'\nGlobal entanglement F0 (χ=1): {F0:.8f}')
    print(f'First layer NLF/F0: {NLF[0]/F0:.8f}')
    print(f'Final NLF/F0: {NLF[-1]/F0:.8f}')
    
    # 4. plot results
    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    plt.plot(range(1, len(NLF)+1), NLF, 'b-o', linewidth=2, markersize=6, 
             label=f'Heisenberg N={N}, χ_max={chi_max}')
    plt.axhline(y=F0, color='r', linestyle='--', linewidth=2, 
                label=f'Global entanglement F0={F0:.4f}')
    
    plt.legend(fontsize=12)
    plt.title('Negative Log Fidelity (NLF) vs Circuit Layers', fontsize=14)
    plt.xlabel('Number of Layers D', fontsize=12)
    plt.ylabel('NLF per site', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    
    # save image
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    plt.savefig(f'./outputs/NLF_Heisenberg_N{N}_D{D_max}_chi{chi_max}.png', dpi=300)
    print(f"\nResult image has been saved to: ./outputs/NLF_Heisenberg_N{N}_D{D_max}_chi{chi_max}.png")

    # plt.show()
