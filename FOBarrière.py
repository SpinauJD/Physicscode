import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

m = 1
a = 1
dt = 0.0001
L = 10
Nx = 40
dx = L / Nx
k0 = 20
Nt = 200000
psi = np.zeros((Nt, Nx), dtype=complex)

listeX = np.arange(0, L, dx)
h = 1j

V = np.zeros(Nx)
V0 = 5
for i in range(Nx):
    if 1.5 <= listeX[i] <= 2.5:
        V[i] = V0

x0 = L / 2
for i in range(0, Nx - 1):
    psi[0, i] = (2*a/np.pi)**(1/4)*np.exp(-a * ((i * dx - x0)**2)) * np.exp(1j * k0 * i * dx)

psi[:, 0] = 0
psi[:, Nx - 1] = 0

for t in range(0, Nt - 1):
    for x in range(1, Nx - 1):
        psi[t+1, x] = psi[t, x] + (h / 2 * m) * (dt / dx**2) * (psi[t, x+1] - 2 * psi[t, x] + psi[t, x-1]) - (h * dt * V[x] * psi[t, x])


# Configuration de la figure avec 2 sous-graphiques
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Configuration du premier graphique (|psi|^2)
ax1.set_xlim(0, L)
ax1.set_ylim(0, np.max(np.abs(psi)**2) * 1.1)  
ax1.axvspan(1.5, 2.5, color='purple', alpha=0.3, label='Potentiel V')
line1, = ax1.plot([], [], lw=2, color='blue', label='|psi|^2')
ax1.set_title("Densité de probabilité au cours du temps")
ax1.legend(loc="upper right")


# Configuration du second graphique (Re(psi))
ax2.set_xlim(0, L)
ax2.set_ylim(-np.max(np.abs(psi)) * 1.1, np.max(np.abs(psi)) * 1.1)
ax2.axvspan(1.5, 2.5, color='purple', alpha=0.3, label='Potentiel V')
line2, = ax2.plot([], [], lw=2, color='orange', label='Re(psi)')
ax2.set_title("Partie réelle de la fonction d'onde au cours du temps")
ax2.legend(loc="upper right")


step = 250  
frames_count = Nt // step

# Fonctions requises par FuncAnimation
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def animate(frame):
    t = frame * step  
    line1.set_data(listeX, np.abs(psi[t, :])**2)
    line2.set_data(listeX, np.real(psi[t, :]))
    return line1, line2

# Création de l'animation
anim = animation.FuncAnimation(fig, animate, init_func=init, 
                               frames=frames_count, interval=20, blit=True)

plt.tight_layout()
plt.show()
