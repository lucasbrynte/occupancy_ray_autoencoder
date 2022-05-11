import torch
from unbalancedot.functional import regularized_ot
# from unbalancedot.functional import sinkhorn_divergence
from unbalancedot.entropy import KullbackLeibler
from unbalancedot.sinkhorn import BatchVanillaSinkhorn
from unbalancedot.utils import euclidean_cost


def verify_inputs(a, b, xa, xb):
    assert torch.is_tensor(a)
    assert torch.is_tensor(b)
    assert torch.is_tensor(xa)
    assert torch.is_tensor(xb)

    dtype = a.dtype
    assert a.dtype == dtype
    assert b.dtype == dtype
    assert xa.dtype == dtype
    assert xb.dtype == dtype

    device = a.device
    assert a.device == device
    assert b.device == device
    assert xa.device == device
    assert xb.device == device

    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert len(xa.shape) == 3
    assert len(xb.shape) == 3

    bs = a.shape[0]
    na = a.shape[1]
    nb = b.shape[1]
    x_dim = xa.shape[2]

    assert a.shape == (bs, na)
    assert b.shape == (bs, nb)
    assert xa.shape == (bs, na, x_dim)
    assert xb.shape == (bs, nb, x_dim)

    assert torch.all(a >= 0)
    assert torch.all(b >= 0)


#def calc_gen_KL_pytorch(p, q):
#    # Implementation of the Csiszar-divergence for the KL entropy f(t) = tlog(t) - t + 1.
#    # We are assuming that p & q are sampled on the same grid / consist of dirac masses at corresponding positions.
#
#    # If p=q=0, the result is 0.
#    # If p=0, standard KL is 0, but the result will be sum(q).
#    # q=0 is not allowed, unless also p=0.
#
#    ### NOTE
#    # This implementation does not yet support batched data!
#    ### NOTE
#
#    p_zero = torch.isclose(p, 0)
#    q_zero = torch.isclose(q, 0)
#    assert torch.all(q[~p_zero] > 0) # Wherever p > 0, we must assume that also q > 0, or else the divergence is infinite.
#    KL = torch.sum(p[~p_zero] * torch.log(p[~p_zero] / q[~p_zero]))
#    gen_KL = KL - torch.sum(p) + torch.sum(q)
#    return gen_KL


# Calculate epsilon-regularized OT cost (not a divergence).
def calc_OT_eps_zero_robust_pytorch(
    a, # Shape (bs, na)
    b, # Shape (bs, nb)
    xa, # Shape (bs, na, x_dim)
    xb, # Shape (bs, nb, x_dim)
    solver = None,
    epsilon = 0.1, # = Entropy parameter
    rho = 1., # = Unbalanced KL relaxation parameter
):
    verify_inputs(a, b, xa, xb)
    bs = a.shape[0]
    na = a.shape[1]
    nb = b.shape[1]
    x_dim = xa.shape[2]
    dtype = a.dtype
    device = a.device

    p = 2
    cost = euclidean_cost(p)
    entropy = KullbackLeibler(epsilon, rho)
    if solver is None:
        # Greedy default settings
        solver = BatchVanillaSinkhorn(
            nits = 100,
            nits_grad = 5,
            tol = 1e-3,
            assume_convergence = True,
        )
    #     solver = BatchVanillaSinkhorn(
    #         nits = 5000,
    #         nits_grad = 15,
    #         tol = 1e-8,
    #         assume_convergence = True,
    #     )

    # Build a mask denoting which of the batch samples require special treatment, since either a or b is a null measure.
    null_case_mask = torch.all(a <= 0, dim=1) | torch.all(b <= 0, dim=1)
    assert null_case_mask.shape == (bs,)
    n_null_case = torch.sum(null_case_mask)

    # Allocate a tensor for all results
    OT_eps = torch.empty((bs,), dtype=dtype, device=device)


    ### First handle the standard case
    tmp = regularized_ot(
        a[~null_case_mask, :],
        xa[~null_case_mask, :, :],
        b[~null_case_mask, :],
        xb[~null_case_mask, :, :],
        cost = cost,
        entropy = entropy,
        solver = solver,
    )
    if tmp.shape == (bs-n_null_case, bs-n_null_case):
        # Legacy case, in which case all but the diagonal is nonsense.
        # assert torch.all(tmp == tmp[[0], :])
        tmp = torch.diag(tmp)
    assert tmp.shape == (bs-n_null_case,)
    OT_eps[~null_case_mask] = tmp


    ### Next handle the singular case
    # Note:
    # If any of a and b is everywhere zero, T has to marginalize to 0 as well at the optimal solution, otherwise the KL terms are infinite.
    # https://stats.stackexchange.com/a/362958
    # Consequently, the optimal transport will be T = 0 in this case.
    T_null = torch.zeros((n_null_case, na, nb))

    L_T = torch.zeros((n_null_case,), dtype=dtype, device=device) # Should be L_T = torch.sum(M.reshape(n_null_case, -1) * T_null.reshape(n_null_case, -1), dim=1), which requires M, but we don't need to calculate M, since T_null=0 and the result is trivially 0.

    # The following three lines of explicit calculations would require a function calc_gen_KL_pytorch() supporting batched data:
    # L_entropic_regul = calc_gen_KL_pytorch(T_null, a[null_case_mask, :, None] @ b[null_case_mask, None, :]) # 0 if a=0 or b=0
    # L_soft_margin_constr_a = calc_gen_KL_pytorch(torch.sum(T_null, dim=2), a[null_case_mask, :]) # 0 if a=0, otherwise sum(a)
    # L_soft_margin_constr_b = calc_gen_KL_pytorch(torch.sum(T_null, dim=1), b[null_case_mask, :]) # 0 if b=0

    # Instead, we simplify the calculations and write the explicit results given our knowledge of how the KL Csiszar-divergence behaves:
    L_entropic_regul = torch.zeros((n_null_case,), dtype=dtype, device=device) # p=T=0, and q=0 too. All terms vanish.
    L_soft_margin_constr_a = torch.sum(a[null_case_mask, :], dim=1) # p=T=0, expression reduces to sum(q)
    L_soft_margin_constr_b = torch.sum(b[null_case_mask, :], dim=1) # p=T=0, expression reduces to sum(q)

    # print(L_T, L_entropic_regul, L_soft_margin_constr_a, L_soft_margin_constr_b)
    # print(L_T.dtype, L_entropic_regul.dtype, L_soft_margin_constr_a.dtype, L_soft_margin_constr_b.dtype)
    OT_eps[null_case_mask] = L_T + epsilon*L_entropic_regul + rho*(L_soft_margin_constr_a + L_soft_margin_constr_b)

    return OT_eps


# Calculate Sinkhorn divergence (OT_eps debiased).
def calc_S_eps_zero_robust_pytorch(
    a, # Shape (bs, na)
    b, # Shape (bs, nb)
    xa, # Shape (bs, na, x_dim)
    xb, # Shape (bs, nb, x_dim)
    solver = None,
    epsilon = 0.1, # = Entropy parameter
    rho = 1., # = Unbalanced KL relaxation parameter
):
    verify_inputs(a, b, xa, xb)
    bs = a.shape[0]
    na = a.shape[1]
    nb = b.shape[1]
    x_dim = xa.shape[2]
    dtype = a.dtype
    device = a.device

    p = 2
    cost = euclidean_cost(p)
    entropy = KullbackLeibler(epsilon, rho)
    if solver is None:
        # Greedy default settings
        solver = BatchVanillaSinkhorn(
            nits = 100,
            nits_grad = 5,
            tol = 1e-3,
            assume_convergence = True,
        )
    #     solver = BatchVanillaSinkhorn(
    #         nits = 5000,
    #         nits_grad = 15,
    #         tol = 1e-8,
    #         assume_convergence = True,
    #     )

    # We call the OT_eps function from before 3 times.
    # Note that if a=0, both OT_eps_ab and OT_eps_aa will be trivial to compute, and the corresponding calls will be very fast. (Equivalently for b=0).
    OT_eps_ab = calc_OT_eps_zero_robust_pytorch(a, b, xa, xb, solver=solver, epsilon=epsilon, rho=rho)
    OT_eps_aa = calc_OT_eps_zero_robust_pytorch(a, a, xa, xa, solver=solver, epsilon=epsilon, rho=rho)
    OT_eps_bb = calc_OT_eps_zero_robust_pytorch(b, b, xb, xb, solver=solver, epsilon=epsilon, rho=rho)

    ma, mb = a.sum(dim=1), b.sum(dim=1) # Computation assumes equidistant grids with sampling interval 1

    assert OT_eps_ab.shape == (bs,)
    assert OT_eps_aa.shape == (bs,)
    assert OT_eps_bb.shape == (bs,)
    assert ma.shape == (bs,)
    assert mb.shape == (bs,)

    S_eps = OT_eps_ab - 0.5*OT_eps_aa - 0.5*OT_eps_bb + 0.5*epsilon*(ma - mb)**2

    return S_eps
