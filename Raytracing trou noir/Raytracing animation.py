import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#parametres
M=1
r_hori=2*M
r_photon=3*M
bc=3*M*np.sqrt(3)

def traj_photon(lambda_param,Y,M,L):
    r,vr,phi=Y
    if r<=r_hori:
        return [0,0,0]
    
    dr=vr 
    dvr=(L**2/r**3)*(1-(3*M/r))
    dphi=L/r**2

    return [dr,dvr,dphi]

param_impact=np.linspace(-15,15,100)

t_eval = np.linspace(0, 70, 300)

trajectoires = []

for b in param_impact:
    x0=-20
    y0=b

    r0=np.sqrt(x0**2+y0**2)
    phi0=np.arctan2(y0,x0)

    L=-y0
    vr0=x0/r0

    Y0 = [r0, vr0, phi0]

    #conditions d'arret
    def capture_horizon(lambda_param,Y,M,L):
        return Y[0]-r_hori

    capture_horizon.terminal = True  #Arrete quand r=r_hori


    def escape_horizon(lambda_param,Y,M,L):
        return Y[0]-25

    escape_horizon.terminal = True  #Arrete quand r=25


    #Intervalle lambda
    lambda_span = (0, 70)

    #Résolution trajectoire
    sol = solve_ivp(               #ordre : (fonction,intervalle,etat initial...)
        traj_photon,
        lambda_span,
        Y0,
        args=(M, L),               #Arguments à passer à traj_photon (solve_ivp en envoit que 2)
        events=[capture_horizon, escape_horizon],
        max_step=0.1,              
        method="RK45",
    )

    #Résultat : sol.y contient [r(lambda), vr(lambda), phi(lambda)]
    r_sol = sol.y[0]
    phi_sol = sol.y[2]

    #Conversion en cartesien
    x=r_sol*np.cos(phi_sol)
    y=r_sol*np.sin(phi_sol)

    couleur= 'red' if abs(b) < bc else 'blue'
    trajectoires.append((x, y, couleur))

# --- 3. Configuration de la figure ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-15, 15)
ax.set_ylim(-10, 10)
ax.set_aspect("equal")
ax.set_title("Animation du Raytracing 2D autour d'un trou noir")

# Dessin du trou noir et de la sphère de photons
Trou_noir = plt.Circle((0, 0), r_hori, color="black", zorder=5)
Sphère_photons = plt.Circle(
    (0, 0), r_photon, fill=False, linestyle="--", color="orange"
)
ax.add_patch(Trou_noir)
ax.add_patch(Sphère_photons)

# Création des éléments graphiques vides pour l'animation
lignes = []
points = []
for _, _, color in trajectoires:
    (line,) = ax.plot([], [], color=color, lw=0.8, alpha=0.6)
    (point,) = ax.plot([], [], marker="o", color=color, markersize=3)
    lignes.append(line)
    points.append(point)


# --- 4. Fonctions pour l'animation ---
def init():
    for line, point in zip(lignes, points):
        line.set_data([], [])
        point.set_data([], [])
    return lignes + points


def update(frame):
    for i, (x, y, _) in enumerate(trajectoires):
        # On affiche la trajectoire parcourue jusqu'à l'étape 'frame'
        idx = min(frame, len(x) - 1)
        lignes[i].set_data(x[: idx + 1], y[: idx + 1])
        # On place un point à la tête du photon
        points[i].set_data([x[idx]], [y[idx]])
    return lignes + points


# --- 5. Lancement de l'animation ---
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(t_eval),
    init_func=init,
    interval=30,
    blit=True,
    repeat=True,
)
ani.save("raytracingtrounoir.gif",writer='pillow',fps=30)
plt.show()
