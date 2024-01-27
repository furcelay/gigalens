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
        data = ", ".join([", ".join([self._print(j) for j in i]) for i in expr.tolist()])
        return data


class ListJAXPrinter(JaxPrinter):
    def _print_MatrixBase(self, expr):
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
