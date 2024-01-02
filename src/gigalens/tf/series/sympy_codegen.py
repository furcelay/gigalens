import sympy as sp
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.utilities.lambdify import _TensorflowEvaluatorPrinter


ORDER = 5


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


def sympy_derivs(expr, var, order):
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


def tf_print(fn_names, exprs, args, cse=sp.cse, graph=True, jit=False):
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

    for name, expr in zip(fn_names, exprs):
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

    func_list_str = "\n\n" + "deflection_fns = [" + ", ".join(fn_names) + "]" + "\n"
    return impstr + funcstr + func_list_str


if __name__ == "__main__":
    x, y, e, r_core, r_cut = sp.symbols('x y e r_core r_cut')
    alpha_xy = sympy_deflection(x, y, e, r_core, r_cut)
    alpha_xy_derivs = sympy_derivs(alpha_xy, r_cut, ORDER)
    alpha_xy_tf = tf_print(
        [f"deflection_{i}" for i in range(ORDER + 1)],
        alpha_xy_derivs,
        [x, y, e, r_core, r_cut]
    )
    with open('dpie_deflection_series.py', 'w') as pyfile:
        pyfile.write(alpha_xy_tf)
