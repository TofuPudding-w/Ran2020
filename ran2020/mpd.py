import numpy as np
import scipy.linalg as LA
from ncon import ncon

def multi_kron(matrices):
    '''
    calculate multi matrices' kronecker
    '''
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result

def canonical_compress_mps(mps_list, max_chi):
    """
    standard dual-direction scan MPS, compress and left canonicalize function
    three steps: 
        R->L no truncation QR 
        L->R svd truncation
        R->L no truncation QR
    output left canonicalized MPS, bundary dimension 1
    ---------------------------------------
    parameters:
        mps_list: input MPS tensor list, shape(L, d_phys, R)
        max_chi: max virtual dimension
    ---------------------------------------
    outputs:
        new_mps: left canonicalized and compressed MPS
    """
    N = len(mps_list)
    new_mps = [t.copy() for t in mps_list]
    
    # Step 1: left to right no truncation QR scan, push the canonical center to the right side
    for i in range(N - 1, 0, -1):
        T = new_mps[i]
        L, d_dim, R = T.shape
        M = T.reshape(L, d_dim * R)
        Q, R_mat = LA.qr(M.T, mode='economic')
        K = Q.shape[1]
        new_mps[i] = Q.T.reshape(K, d_dim, R)
        left_T = new_mps[i - 1]
        new_mps[i - 1] = np.tensordot(left_T, R_mat.T, axes=([2], [0]))
    
    # Step 2: left to right SVD scan, truncate to max_chi
    for i in range(N - 1):
        T = new_mps[i]
        L, d_dim, R = T.shape
        M = T.reshape(L * d_dim, R)
        U, S, Vh = LA.svd(M, full_matrices=False)
        k = min(len(S), max_chi)
        new_mps[i] = U[:, :k].reshape(L, d_dim, k)
        SVh = np.diag(S[:k]) @ Vh[:k, :]
        right_T = new_mps[i + 1]
        new_mps[i + 1] = np.tensordot(SVh, right_T, axes=([1], [0]))
    
    # Step 3: right to left no truncation QR scan, restore to left canonical state
    for i in range(N - 1, 0, -1):
        T = new_mps[i]
        L, d_dim, R = T.shape
        M = T.reshape(L, d_dim * R)
        Q, R_mat = LA.qr(M.T, mode='economic')
        K = Q.shape[1]
        new_mps[i] = Q.T.reshape(K, d_dim, R)
        new_mps[i-1] = np.tensordot(new_mps[i-1], R_mat.T, axes=([2], [0]))
    
    # global normalize
    norm = np.linalg.norm(new_mps[0])
    if norm > 1e-12:
        new_mps[0] /= norm
    
    return new_mps

def get_orthonormal_complement(V, D):
    """
    get the orthonormal complement of equidistant matrix, 
    complete to D dimention unitary matrix
    """
    m = V.shape[0]
    if m == D:
        return V
    # calculate null space and transpose conjugate, guarantee the orthogonality
    null_space = LA.null_space(V).conj().T
    # print(V.shape, null_space.shape)
    return np.vstack((V, null_space))

def disentangler_from_mps(mps, d):
    """
    build a single layer of MPD according to the right orthonormal MPS(χ=d)
    output G shape (d, d, d, d)
    -----------------------------------
    parameters:
        mps: right orthonormal MPS, all virtual bond dimension χ=d
        d: physic dimension (d=2)
    -----------------------------------
    outputs:
        gates: disentangle the list of gates, each gate is a 4 dimension tensor (vL, d_out, d_in, vR)
    """
    N = len(mps)
    gates = []
    
    for i, A in enumerate(mps):
        L, d_in, R = A.shape
        assert d_in == d, f"The physic dimension isn't match: expected {d}, practically {d_in}"
        # d*d
        A_pad = np.zeros((L, d, d), dtype=complex)
        A_pad[:, :d_in, :R] = A
        V = A_pad.reshape(L, d*d)

        # complete U as unitary matrix
        U_mat = get_orthonormal_complement(V, d * d)
        # reshape to 4 dimension gate tensor：(d_out, d_virtual, d_phys, d_virtual_out)
        G = U_mat.conj()
        # (d_out, vL, d_in, vR) -> (vL, d_out, d_in, vR)
        G = U_mat.reshape(d, d, d, d).transpose(1, 0, 2, 3)
        
        assert np.allclose(G.conj().T @ G, np.eye(d*d), atol=1e-10)

        # deal with system boundary's virtual leg
        if i == 0: 
            G = G[0:1, :, :, :]
        if i == N - 1: 
            G = G[:, :, :, 0:1]
        gates.append(G)
    
    return gates

def apply_disentangler(psi_tensors, gates, d):
    """
    apply one layer of MPD gates to targeted MPS
    -----------------------------
    parameters:
        psi_tensors: list of input MPS tensor
        gates: list of disentangler gates
    -----------------------------
    outputs:
        new_tensors: MPS after applying, the virtual bond will explode to the d times of origin
    """
    new_mps = []
    for W, A in zip(gates, psi_tensors):
        # W: (vL, d_out, d_in, vR), A: (L, d_in, R)
        T = np.tensordot(W, A, axes=([2], [1]))
        # ((vL*L), d_out, (vR*R))
        T = T.transpose(0, 3, 1, 2, 4)
        vL, L, d_out, vR, R = T.shape
        T = T.reshape(vL * L, d_out, vR * R)
        new_mps.append(T)
    return new_mps

# def fidelity_with_vacuum(psi_tensors):
#     N = len(psi_tensors)

#     R_env = np.ones(1, dtype=complex)
    
#     for i in range(N-1, -1, -1):
#         T = psi_tensors[i]
#         T_proj = T[:, 0, :]  # (aL, aR)
#         R_env = np.tensordot(T_proj, R_env, axes=([1], [0]))
    
#     overlap = R_env[0]
#     return np.abs(overlap) ** 2

def get_all_zero_amplitude(mps):
    """
    calculate the inner product amplitude of MPS and |000...00>
    """
    vec = np.eye(1)
    for T in mps:
        M_zero = T[:, 0, :]
        # print(vec.shape, M_zero.shape)
        vec = vec @ M_zero
    return np.abs(vec[0, 0])

def mps_overlap(mps1, mps2):
    """
    calculate the inner product of two MPS < mps1 | mps2 >
    """
    N = len(mps1)
    E = np.eye(1)
    for i in range(N):
        A1 = mps1[i].conj()
        A2 = mps2[i]
        tmp = np.tensordot(E, A1, axes=([0], [0]))
        E = np.tensordot(tmp, A2, axes=([0, 1], [0, 1]))
    return np.abs(E[0, 0])

def encoding_algorithm(target_mps_tensors, D_max, d, N, chi_max):
    """
    complete MPS disentangle algorithm
    iterate and build multi-layer MPD, disentangle targeted MPS to product state
    -------------------------------------
    parameters:
        target_mps_tensors: target MPS (from DMRG)
        D_max: max circuit layer
        d: physic dimension
        N: system's site number
        chi_max: the max dimension of virtual bond
    -------------------------------------
    outputs:
        all_gates: list of all layers' disentangle gate
        NLF: list of every layer's negative logarithm fidelity
    """
    # initialize: right canonicalize targeted MPS
    psi_current = canonical_compress_mps(target_mps_tensors, max_chi=chi_max)
    all_gates = []
    NLF = []
    
    print(f"\n===== Start MPS Encoding Algorithm =====")
    print(f"System size: N={N}, Physics dimension: d={d}")
    print(f"Max circuit layer: D_max={D_max}, Max virtual bond: χ_max={chi_max}")
    print(f"Initialize the global tanglement F0: {-np.log(get_all_zero_amplitude(canonical_compress_mps(psi_current, 1)**2) + 1e-30)/N:.6f}\n")
    
    for layer in range(D_max):

        # Step 1: truncate current MPS to χ=d, get the optimal approximate
        psi_truncated = canonical_compress_mps(psi_current, max_chi=d)
        
        # Step 2: build single layer disentangler based on truncated MPS
        gates = disentangler_from_mps(psi_truncated, d)
        all_gates.append(gates)
        
        # Step 3: apply disentagler to current MPS
        psi_new = apply_disentangler(psi_current, gates, d)
        
        # Step 4: compress MPS to chi_max
        psi_current = canonical_compress_mps(psi_new, max_chi=chi_max)
        
        # Step 5: calculate the negative logarithm fidelity(NLF) of current layer
        fid = get_all_zero_amplitude(psi_current)
        fid = fid ** 2  # fidelity**2 = the modulus of inner product
        nlf = -np.log(fid + 1e-30) / N
        NLF.append(nlf)
        
        print(f"Layer {layer+1:2d}: NLF = {nlf:.8f}, Tensor shape of middle tensor: {psi_current[N//2].shape}")
    
    print(f"\n===== Encoding Finished =====")
    print(f"Final NLF: {NLF[-1]:.8f}")
    print(f"{NLF[0]/NLF[-1]:.2f} times lower than initial tanglement")
    
    return all_gates, NLF
