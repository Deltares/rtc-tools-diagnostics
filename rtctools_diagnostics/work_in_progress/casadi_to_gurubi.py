""" The latest version of our conversion file. Needs to be updated, refactored,
and validated and then integrated into this repo."""

import pickle
import glob, os, shutil
import casadi as ca
import numpy as np
import textwrap

ps = sorted(glob.glob("nlp*.pickle"))

with open(ps[-1], "rb") as f:
    d = pickle.load(f)

pickleid = os.path.basename(ps[-1]).split(".")[0].rsplit("_", 1)[1]

indices = d["indices"][0]
expand_f_g = d["func"]
lbx, ubx, lbg, ubg, x0 = d["other"]
X = ca.SX.sym("X", expand_f_g.nnz_in())
f, g = expand_f_g(X)

in_var = X
out = []
for o in [f, g]:
    Af = ca.Function("Af", [in_var], [ca.jacobian(o, in_var)])
    bf = ca.Function("bf", [in_var], [o])

    A = Af(0)
    A = ca.sparsify(A)

    b = bf(0)
    b = ca.sparsify(b)
    out.append((A, b))

var_names = []
for k, v in indices.items():
    if isinstance(v, int):
        var_names.append("{}__{}".format(k, v))
    elif isinstance(v, np.ndarray):
        for i in range(0, v.shape[0]):
            var_names.append("{}__{}".format(k, i))
    else:
        for i in range(0, v.stop - v.start, 1 if v.step is None else v.step):
            var_names.append("{}__{}".format(k, i))

n_derivatives = expand_f_g.nnz_in() - len(var_names)
for i in range(n_derivatives):
    var_names.append("DERIVATIVE__{}".format(i))


# CPLEX does not like [] in variable names
import re

for i, v in enumerate(var_names):
    v = v.replace("[", "_I")
    v = v.replace("]", "I_")
    var_names[i] = v

# OBJECTIVE
A, b = out[0]
objective = []
ind = np.array(A)[0, :]

for v, c in zip(var_names, ind):
    if c != 0:
        objective.extend(["+" if c > 0 else "-", str(abs(c)), v])

if objective[0] == "-":
    objective[1] = "-" + objective[1]

objective.pop(0)
objective_str = " ".join(objective)
objective_str = "  " + objective_str

# CONSTRAINTS
A, b = out[1]
ca.veccat(*lbg)
lbg = np.array(ca.veccat(*lbg))[:, 0]
ubg = np.array(ca.veccat(*ubg))[:, 0]


A_csc = A.tocsc()
A_coo = A_csc.tocoo()
b = np.array(b)[:, 0]

constraints = [[] for i in range(A.shape[0])]

for i, j, c in zip(A_coo.row, A_coo.col, A_coo.data):
    constraints[i].extend(["+" if c > 0 else "-", str(abs(c)), var_names[j]])

for i in range(len(constraints)):
    cur_constr = constraints[i]
    l, u, b_i = lbg[i], ubg[i], b[i]

    if len(cur_constr) > 1:
        if cur_constr[0] == "-":
            cur_constr[1] = "-" + cur_constr[1]
        cur_constr.pop(0)

    c_str = " ".join(cur_constr)
    if cur_constr == []:
        print("error")
        constraints[i] = "\ "
        continue

    if np.isfinite(l) and np.isfinite(u) and l == u:
        constraints[i] = "{} = {}".format(c_str, l - b_i)
    elif np.isfinite(l):
        constraints[i] = "{} >= {}".format(c_str, l - b_i)
    elif np.isfinite(u):
        constraints[i] = "{} <= {}".format(c_str, u - b_i)
    else:
        # print('should be exception')
        raise Exception(l, b, constraints[i])

constraints_str = "  " + "\n  ".join(constraints)

# Bounds
bounds = []
for v, l, u in zip(var_names, lbx, ubx):
    bounds.append("{} <= {} <= {}".format(l, v, u))
bounds_str = "  " + "\n  ".join(bounds)

with open("myproblem_{}.lp".format(pickleid), "w") as o:
    o.write("Minimize\n")
    for x in textwrap.wrap(
        objective_str, width=255
    ):  # lp-format has max length of 255 chars
        o.write(x + "\n")
    #    o.write(objective_str + "\n")
    o.write("Subject To\n")
    o.write(constraints_str + "\n")
    o.write("Bounds\n")
    o.write(bounds_str + "\n")
    expand_discrete = d[
        "discrete"
    ]  # an array of booleans, in the same order as the variable names
    if any(expand_discrete):
        o.write("General\n")
        whitespace_separated_discrete_var_names = ""
        for i in range(len(var_names)):
            if expand_discrete[i]:
                whitespace_separated_discrete_var_names += var_names[i]
                whitespace_separated_discrete_var_names += " "
        o.write(whitespace_separated_discrete_var_names + "\n")
    o.write("End")

shutil.copy("myproblem_{}.lp".format(pickleid), "myproblem.lp")
nrows = A_coo.shape[0]

ratios = []
minmaxs = []

# shutil.copy("myproblem.lp", r"C:\myproblem.lp")

# for i in range(nrows):
#    d = np.abs(A_coo.getrow(i).data)
#    m, M = min(d), max(d)
#    minmaxs.append((m, M))
#    ratios.append(abs(M/m))

# print(max(ratios))
