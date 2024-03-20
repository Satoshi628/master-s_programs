import torch

from .misc import safe_detach


class ODESolver(object):
    """
    Fixed step size ODE solver
    """
    def __init__(self, func):
        self.func = func

    def integrate(self, x0, times):
        x = x0
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            dx = self._step_fn(t0, x, dt)
            x = x + dx

        return x

    def _step_fn(self, t, x, dt):
        pass


class Midpoint(ODESolver):
    """
    Fixed step size midpoint method
    """
    def __init__(self, func):
        super(Midpoint, self).__init__(func)

    def _step_fn(self, t, x, dt):
        k1 = dt * self.func(t, x)
        k2 = dt * self.func(t + 0.5 * dt, x + 0.5 * k1)
        return k2


class RK4(ODESolver):
    """
    4-th order Runge-Kutta method
    """
    def __init__(self, func):
        super(RK4, self).__init__(func)

    def _step_fn(self, t, x, dt):
        k1 = dt * self.func(t, x)
        k2 = dt * self.func(t + 0.5 * dt, x + 0.5 * k1)
        k3 = dt * self.func(t + 0.5 * dt, x + 0.5 * k2)
        k4 = dt * self.func(t + dt, x + k3)
        dx = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return dx


class AdaptiveODESolver(object):
    """
    Base class for adaptive step size ODE solvers
    """
    def __init__(self, func, rtol, atol):
        self.func = func
        self.atol = atol
        self.rtol = rtol
        self.order = -1
        self.c_t = None
        self.c_x = None
        self.c_err = None

    def integrate(self, x0, times):
        t_start = times[0]
        t_end = times[-1]
        dt = (t_end - t_start) / (len(times) - 1)
        dt_min = torch.abs(dt) * 0.2
        dt_max = torch.abs(dt) * 5.0

        # adaptive integration
        x1 = x0
        t0 = t_start
        t1 = t_start
        while abs(t1 - t_end) > 1.0e-4:
            dx, _ = self._step_fn(t1, x1, dt)
            dt = torch.clamp(torch.abs(dt), dt_min, dt_max) * torch.sign(dt)
            if (t_start - (t1 + dt)) * (t_end - (t1 + dt)) > 0.0:
                dt = t_end - t1

            x0 = x1
            t0 = t1
            x1 = x1 + dx
            t1 = t1 + dt

        # lerp to get value at "t_end"
        slope = (t_end - t0) / (t1 - t0)
        x = x0 + (x1 - x0) * slope

        return x

    def _step_fn(self, t, x, dt):
        k0 = dt * self.func(t, x)
        ks = [k0]
        for i in range(self.order + 1):
            kx = sum([k * c for k, c in zip(ks, self.c_x[i])])
            ki = dt * self.func(t + self.c_t[i] * dt, x + kx)
            ks.append(ki)

        dx = sum([k * c for k, c in zip(ks, self.c_x[-1])])
        x_err = sum([k * c for k, c in zip(ks, self.c_err)])

        etol = self.atol + self.rtol * torch.max(x.abs(), (x + dx).abs())
        err_norm = (x_err / etol).pow(2).mean().sqrt()
        dt_new = dt * (0.5 / err_norm)**(1.0 / self.order)

        return dx, dt_new


class Bosha3(AdaptiveODESolver):
    """
    Adaptive step size Bogacki-Shampine method
    """
    def __init__(self, func, rtol=1.0e-3, atol=1.0e-3):
        super(Bosha3, self).__init__(func, rtol, atol)
        self.order = 3
        self.c_t = [1.0 / 2.0, 3.0 / 4.0, 1.0, 1.0]
        self.c_x = [
            [1.0 / 2.0],
            [0.0, 3.0 / 4.0],
            [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0],
            [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0],
        ]
        self.c_err = [
            2.0 / 9.0 - 7.0 / 24.0,
            1.0 / 3.0 - 1.0 / 4.0,
            4.0 / 9.0 - 1.0 / 3.0,
            0.0 - 1.0 / 8.0,
        ]


class Dopri5(AdaptiveODESolver):
    """
    Adaptive step size Dormand Prince method
    """
    def __init__(self, func, rtol=1.0e-2, atol=1.0e-2):
        super(Dopri5, self).__init__(func, rtol, atol)
        self.order = 5
        self.c_t = [1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0]
        self.c_x = [
            [1.0 / 5.0],
            [3.0 / 40.0, 9.0 / 40.0],
            [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
            [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
            [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
            [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
        ]
        self.c_err = [
            35.0 / 384.0 - 5179.0 / 57600.0,
            0.0 - 0.0,
            500.0 / 1113.0 - 7571.0 / 16695.0,
            125.0 / 192.0 - 393.0 / 640.0,
            -2187.0 / 6784.0 + 92097.0 / 339200.0,
            11.0 / 84.0 - 187.0 / 2100.0,
            0.0 - 1.0 / 40.0,
        ]


SOLVERS = {
    'midpoint': Midpoint,
    'rk4': RK4,
    'bosha3': Bosha3,
    'dopri5': Dopri5,
}


def _to_tuple(x, shapes):
    tensors = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        if shape.numel() > 1:
            t = x[total:next_total].view(*shape)
        else:
            t = x[total:next_total].view(1)
        tensors.append(t)
        total = next_total

    return tuple(tensors)


def _to_flat(x):
    x = [x_.reshape(-1) for x_ in x]
    return torch.cat(x)


def _tuple_func_wrapper(func, shape):
    def _func(t, x):
        x = _to_tuple(x, shape)
        dx = func(t, x)
        dx = _to_flat(dx)
        return dx

    return _func


def odeint(func, x, times, method):
    if isinstance(x, torch.Tensor):
        x = (x, )
    elif not isinstance(x, tuple):
        raise Exception('"odeint" input must be torch.Tensor or tuple')

    shapes = [x_.shape for x_ in x]

    x = _to_flat(x)
    func = _tuple_func_wrapper(func, shapes)
    solver = SOLVERS[method](func)
    x_new = solver.integrate(x, times)
    x_new = _to_tuple(x_new, shapes)
    return x_new


def odeint_adjoint(func, x, times, method):
    params = list(func.parameters())
    params = [p for p in params if p.requires_grad]

    shapes = [x_.shape for x_ in x]
    x = _to_flat(x)
    x_new = OdeIntAdjoint.apply(func, x, shapes, times, method, *params)
    return x_new


def aug_func_wrapper(func, t, states, z_shapes, *params):
    adj_z = states[0]
    z = states[1]
    params = tuple(params)

    with torch.enable_grad():
        t = safe_detach(t)
        z = safe_detach(z)
        z.requires_grad_(True)

        f = _to_flat(func(t, _to_tuple(z, z_shapes)))
        vjp_z, *vjp_params = torch.autograd.grad(f, (z, ) + params,
                                                 grad_outputs=-1.0 * adj_z,
                                                 retain_graph=True,
                                                 allow_unused=True)

    vjp_params = [
        torch.zeros_like(p) if vjp_p is None else vjp_p for p, vjp_p in zip(params, vjp_params)
    ]

    return (vjp_z, f, *vjp_params)


class OdeIntAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, states, shapes, times, method, *params):
        ctx.func = func
        ctx.shapes = shapes
        ctx.method = method
        states = _to_tuple(states, shapes)

        with torch.no_grad():
            z, log_df_dz = odeint(func, states, times, method)

        new_states = _to_flat((z, log_df_dz))
        ctx.save_for_backward(new_states, times, *params)

        return safe_detach(z), safe_detach(log_df_dz)

    @staticmethod
    def backward(ctx, dL_dz, dL_dlogpz):
        func = ctx.func
        shapes = ctx.shapes
        method = ctx.method
        z, times, *params = ctx.saved_tensors
        adj_z = _to_flat((dL_dz, dL_dlogpz))

        with torch.no_grad():
            zero_params = [torch.zeros_like(p) for p in params]
            aug_states = (adj_z, z, *zero_params)
            aug_func = lambda t, z, func=func, shapes=shapes, params=params: aug_func_wrapper(
                func, t, z, shapes, *params)
            aug_states = odeint(aug_func, aug_states, reversed(times), method)

        adj_states = aug_states[0]
        adj_params = aug_states[2:]

        return (None, adj_states, None, None, None, *adj_params)
