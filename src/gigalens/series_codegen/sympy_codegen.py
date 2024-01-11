import sympy as sp
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.numpy import JaxPrinter
from sympy.utilities.lambdify import _EvaluatorPrinter, _TensorflowEvaluatorPrinter
from gigalens.series_codegen import profiles
from os import path


ORDER = 5
BACKEND = "jax"  # "tensorflow"  # jax or tensorflow
PROFILE = "dpie"
OUTDIRS = {
    "jax": path.join('..', 'jax', 'series', 'profiles'),
    "tensorflow": path.join('..', 'tf', 'series', 'profiles')
}
PROFILES = {
    "dpie": profiles.DPIE(),
}


def sympy_deflection(x, y, e, r_core, r_cut):
    scale = r_cut / (r_cut - r_core)
    q = (1 - e) / (1 + e)
    sqe = sp.sqrt(e)
    rem2 = x ** 2 / (1. + e) ** 2 + y ** 2 / (1. - e) ** 2

    zci_re = 0
    zci_im = -0.5 * (1. - e ** 2) / sqe

    # r_core: zsi_rc = (a + bi)/(c + di)
    znum_rc_re = q * x  # a
    znum_rc_im = 2. * sqe * sp.sqrt(r_core ** 2 + rem2) - y / q  # b
    zden_rc_re = x  # c
    zden_rc_im = 2. * r_core * sqe - y  # d

    # r_cut: zsi_rcut = (a + ei)/(c + fi)
    # znum_rcut_re = znum_rc_re  # a
    znum_rcut_im = 2. * sqe * sp.sqrt(r_cut ** 2 + rem2) - y / q  # e
    # zden_rcut_re = zden_rc_re  # c
    zden_rcut_im = 2. * r_cut * sqe - y  # f

    aa = (znum_rc_re * zden_rc_re - znum_rc_im * zden_rcut_im)
    bb = (znum_rc_re * zden_rcut_im + znum_rc_im * zden_rc_re)
    cc = (znum_rc_re * zden_rc_re - zden_rc_im * znum_rcut_im)
    dd = (znum_rc_re * zden_rc_im + zden_rc_re * znum_rcut_im)

    # zis_rc / zis_rcut = ((aa * cc + bb * dd) / norm) + ((bb * cc - aa * dd) / norm) * I
    # zis_rc / zis_rcut = aaa + bbb * I
    norm = (cc ** 2 + dd ** 2)
    aaa = (aa * cc + bb * dd) / norm
    bbb = (bb * cc - aa * dd) / norm

    # compute the zr = log(zis_rc / zis_rcut) = log(aaa + bbb * I)
    norm2 = aaa ** 2 + bbb ** 2
    zr_re = sp.log(sp.sqrt(norm2))
    zr_im = sp.atan2(bbb, aaa)

    # now compute final result: zres = zci * log(zr)
    zres_re = zci_re * zr_re - zci_im * zr_im
    zres_im = zci_im * zr_re + zci_re * zr_im
    return scale * sp.Matrix([zres_re, zres_im])


def sympy_series(expr, var, order):
    derivs = []
    expr_d = expr
    derivs.append(expr_d)
    for i in range(order):
        expr_d = expr_d.diff(var)
        derivs.append(expr_d)
    return derivs


class ListTFPrinter(TensorflowPrinter):
    def _print_MatrixBase(self, expr):
        tensorflow_f = ""
        data = ", ".join([", ".join([self._print(j) for j in i]) for i in expr.tolist()])
        return data


class ListJAXPrinter(JaxPrinter):
    def _print_MatrixBase(self, expr):
        tensorflow_f = ""
        data = ", ".join([", ".join([self._print(j) for j in i]) for i in expr.tolist()])
        return data


def tf_print(fn_names, exprs, args, cse=sp.cse, graph=True, jit=False):
    deriv_names, hessian_names = fn_names
    deriv_exprs, hessian_exprs = exprs

    printer = ListTFPrinter(
        {'fully_qualified_modules': False, 'inline': True}
    )
    func_printer = _TensorflowEvaluatorPrinter(printer=printer, dummify=False)

    decorator = ""
    if graph:
        decorator += "@tensorflow.function"
        if jit:
            decorator += "(jit_compile=True)"

    funcstr = ""

    for name, expr in zip(deriv_names + hessian_names, deriv_exprs + hessian_exprs):
        funcstr += "\n\n"
        if graph:
            funcstr += decorator + "\n"
        cses, expr = cse(expr)
        funcstr += func_printer.doprint(name, args, expr, cses=cses)

    impstr = ""
    if graph:
        impstr += "import tensorflow\n"
    for mod, keys in (getattr(printer, 'module_imports', None) or {}).items():
        for k in keys:
            impstr += f"from {mod} import {k}\n"

    deriv_list_str = "\n\n" + "deriv_fns = [" + ", ".join(deriv_names) + "]" + "\n"
    hessian_list_str = "\n\n" + "hessian_fns = [" + ", ".join(hessian_names) + "]" + "\n"
    return impstr + funcstr + deriv_list_str + hessian_list_str


def jax_print(fn_names, exprs, args, cse=sp.cse, jit=True):
    deriv_names, hessian_names = fn_names
    deriv_exprs, hessian_exprs = exprs

    printer = ListJAXPrinter(
        {'fully_qualified_modules': False, 'inline': True}
    )
    func_printer = _EvaluatorPrinter(printer=printer, dummify=False)

    decorator = ""
    if jit:
        decorator += "@functools.partial(jit, static_argnums=(0,))"

    funcstr = ""

    for name, expr in zip(deriv_names + hessian_names, deriv_exprs + hessian_exprs):
        funcstr += "\n\n"
        if jit:
            funcstr += decorator + "\n"
        cses, expr = cse(expr)
        funcstr += func_printer.doprint(name, args, expr, cses=cses)

    impstr = ""
    if jit:
        impstr += "import functools\nfrom jax import jit\n"
    for mod, keys in (getattr(printer, 'module_imports', None) or {}).items():
        for k in keys:
            impstr += f"from {mod} import {k}\n"

    deriv_list_str = "\n\n" + "deriv_fns = [" + ", ".join(deriv_names) + "]" + "\n"
    hessian_list_str = "\n" + "hessian_fns = [" + ", ".join(hessian_names) + "]" + "\n"
    return impstr + funcstr + deriv_list_str + hessian_list_str


PRINTERS = {
    "jax": jax_print,
    "tensorflow": tf_print
}


if __name__ == "__main__":

    outdir = OUTDIRS[BACKEND]
    profile = PROFILES[PROFILE]

    deriv = profile.deriv(*profile.args)
    deriv_series = sympy_series(deriv, profile.series_var, ORDER)
    hessian = profile.hessian(*profile.args)
    hessian_series = sympy_series(hessian, profile.series_var, ORDER)

    print_fn = PRINTERS[BACKEND]

    alpha_xy_str = print_fn(
        [[f"deriv_{i}" for i in range(ORDER + 1)],
         [f"hessian_{i}" for i in range(ORDER + 1)]],
        [deriv_series, hessian_series],
        profile.args
    )

    outfile = path.join(OUTDIRS[BACKEND], f"{PROFILE.lower()}.py")

    with open(outfile, 'w') as pyfile:
        pyfile.write(alpha_xy_str)
