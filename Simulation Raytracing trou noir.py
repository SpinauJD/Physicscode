import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

param_impact=np.linspace(-15,15,500)

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
    x_sol=r_sol*np.cos(phi_sol)
    y_sol=r_sol*np.sin(phi_sol)

    couleur= 'red' if abs(b) < bc else 'blue'
    plt.plot(x_sol,y_sol,color=couleur,lw=0.5)


plt.plot([],[],color='red',label="Photons capturés ($|b| < b_c$)")
plt.plot([],[],color='blue',label="Photons déviés ($|b| > b_c$)")

#Trou noir
ax=plt.gca()
Trou_noir=plt.Circle((0,0),r_hori,color="black",label="Trou noir")
ax.add_patch(Trou_noir)
ax.set_aspect('equal')

#Sphère de photons
Sphère_photons=plt.Circle((0,0),r_photon,fill=False,linestyle='--',color="orange",label="Sphère de photons")
ax.add_patch(Sphère_photons)

plt.xlim(-15,15)
plt.ylim(-10,10)
plt.title("Raytracing 2D autour d'un trou noir")
plt.legend(loc="upper right")
plt.show()
