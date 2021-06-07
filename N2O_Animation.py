import csv
from matplotlib import pyplot as plt, animation
from matplotlib import cm
import numpy as np

# Setup data for N2O
N2O_time = dict([("Re-1", []), ("Re-2", []), ("Re-3", []), ("Re-4", []), ("Re-5", [])])
N2O_yield = dict([("Re-1", []), ("Re-2", []), ("Re-3", []), ("Re-4", []), ("Re-5", [])])

with open("data/N2O_Photoredox_N2O_kinetics.csv", "r") as data_file:
    data = csv.reader(data_file, delimiter=",")
    next(data)
    next(data)
    next(data)
    for row in data:
        N2O_time["Re-1"].append(row[0])
        N2O_yield["Re-1"].append(row[1])
        N2O_time["Re-2"].append(row[2])
        N2O_yield["Re-2"].append(row[3])
        N2O_time["Re-3"].append(row[4])
        N2O_yield["Re-3"].append(row[5])
        N2O_time["Re-4"].append(row[6])
        N2O_yield["Re-4"].append(row[7])
        N2O_time["Re-5"].append(row[8])
        N2O_yield["Re-5"].append(row[9])

# Cast str to float
N2O_yield = {j: [float(i) for i in N2O_yield[j] if i] for j in N2O_yield}
N2O_time = {j: [float(i) for i in N2O_time[j] if i] for j in N2O_time}


# Setup Re fit functions
def fit_re1(t):
    return (
        52.89551 + (2.134741 - 52.89551) / (1 + (t / 60741890) ** 0.9826197) ** 3977385
    )


def fit_re2(t):
    return 78.38749 + (1.406563 - 78.38749) / (1 + (t / 4.148584) ** 0.8541969)


def fit_re3(t):
    return 96.05159 + (-1.200536 - 96.05159) / (1 + (t / 5.49206) ** 0.7443591)


def fit_re4(t):
    return (
        2958533
        + (1.89292 - 2958533) / (1 + (t / 1.901817) ** 0.5896842) ** 0.000006495978
    )


def fit_re5(t):
    return 8402670 + (
        3.051329 - 8402670 / (1 + (t / 0.7059825) ** 1.331645) ** 0.0000007138259
    )


# Setup animation functions
def init():
    line.set_data([], [])
    return (line,)


def update(i):
    x = i * 0.01
    y = fit_re1(x)
    line.set_data(x, y)
    return (line,)

# https://www.idpoisson.fr/perrollaz/Animation.html

fig = plt.figure()
(line,) = plt.plot([], [])
plt.xlim = (0, 60)
plt.ylim = (0, 100)
colormap = cm.get_cmap("tab20")

x = np.linspace(0, 60, 1000)
y1 = [fit_re1(t) for t in x]
y2 = [fit_re2(t) for t in x]
y3 = [fit_re3(t) for t in x]
y4 = [fit_re4(t) for t in x]
y5 = [fit_re5(t) for t in x]

ani = animation.FuncAnimation(fig, update, init_func=init, frames=200)
plt.show()

# plt.plot(
#     N2O_time["Re-1"],
#     N2O_yield["Re-1"],
#     ".",
#     color=colormap(0),
#     markersize=10,
#     label="Re-1",
# )
# plt.plot(
#     N2O_time["Re-2"],
#     N2O_yield["Re-2"],
#     ".",
#     color=colormap(0.1),
#     markersize=10,
#     label="Re-2",
# )
# plt.plot(
#     N2O_time["Re-3"],
#     N2O_yield["Re-3"],
#     ".",
#     color=colormap(0.9),
#     markersize=10,
#     label="Re-3",
# )
# plt.plot(
#     N2O_time["Re-4"],
#     N2O_yield["Re-4"],
#     ".",
#     color=colormap(0.3),
#     markersize=10,
#     label="Re-4",
# )
# plt.plot(
#     N2O_time["Re-5"],
#     N2O_yield["Re-5"],
#     ".",
#     color=colormap(0.4),
#     markersize=10,
#     label="Re-5",
# )
# plt.plot(x, y1, "--", color=colormap(0.05))
# plt.plot(x, y2, "--", color=colormap(0.15))
# plt.plot(x, y3, "--", color=colormap(0.95))
# plt.plot(x, y4, "--", color=colormap(0.35))
# plt.plot(x, y5, "--", color=colormap(0.45))

# plt.legend(prop={'weight':'bold', 'family':'Calibri', 'size':'large'})
# plt.savefig("data/N2O_Photoredox_N2O_kinetics.png", dpi=300)
