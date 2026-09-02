import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Paramètres
s=1
e=1
E=48*e
dt=0.01
t_max=10
steps= int(t_max/dt)
N_part=50
T_target=2
L=4
V=(2*L)**2

#tableaux traj
pos=np.zeros((steps,N_part,2))
vel=np.zeros((steps,N_part,2))
en_tot=np.zeros(steps)
press=np.zeros(steps)
fp = np.zeros(steps)

#Conditions initiales
pos[0]=np.random.uniform(-L,L,(N_part,2))
vel[0]=np.random.uniform(-L,L,(N_part,2))

#boucle traj
for n in range (steps-1):
    acc=np.zeros((N_part,2))
    Ep=0
    P=0
    for i in range (N_part):
        for j in range (N_part):
            if i!=j:
                vec_r=pos[n,i]-pos[n,j]
                dist=np.sqrt(np.sum(vec_r**2))
                dist=max(dist,0.8)
                f=(E/dist**2)*((s/dist)**12-(1/2)*(s/dist)**6)*vec_r
                acc[i]+=(E/dist**2)*((s/dist)**12-(1/2)*(s/dist)**6)*vec_r
                Ep +=4*e*((s/dist)**12-(s/dist)**6)
        P+=np.dot(vec_r,f)

    #actu traj
    vel[n+1]=vel[n]+acc*dt
    if n%10 == 0:
        #Calcul T
        Ec=0.5*np.sum(vel[n+1]**2)
        T=Ec/N_part

        #Facteur corr   
        if T>0:
            factor=np.sqrt(T_target/T)
            vel[n+1]*=factor
    pos[n+1]=pos[n]+vel[n+1]*dt
    en_tot[n]=Ec+Ep
    press[n]=(N_part*T)/V+(1/(2*V))*P
    p_cum=np.cumsum(press)
    if n!=0:
        fp[n]=p_cum[n]/n
    
    #part hors limites
    hit_left=pos[n+1, :,0]<-L
    hit_right=pos[n+1, :,0]>L
    hit_lefty=pos[n+1, :,1]<-L
    hit_righty=pos[n+1, :,1]>L

    #inverse vel et pos
    vel[n+1,hit_left | hit_right, 0]*=-1
    vel[n+1,hit_lefty | hit_righty, 1]*=-1
    pos[n+1, :,0]=np.clip(pos[n+1, :,0],-L,L)
    pos[n+1, :,1]=np.clip(pos[n+1, :,1],-L,L)

#graph energetique
plt.figure()
plt.plot(en_tot[:-1],color='blue')  # On ignore le dernier point qui n'est pas calculé
plt.title("Conservation de l'énergie totale")
plt.xlabel("Temps (pas)")
plt.ylabel("Énergie (E_c + E_p)")
plt.show()

#graph de pression
plt.figure()
plt.plot(fp[:-1],color='red')  # On ignore le dernier point qui n'est pas calculé
plt.title("Pression")
plt.xlabel("Temps (pas)")
plt.ylabel("P")
plt.show()

#animation
fig,ax=plt.subplots(figsize=(10,7))
ax.set_xlim(-(L+2),L+2)
ax.set_ylim(-(L+2),L+2)
ax.set_title("Animation particules")

point, =ax.plot([],[], 'ro')
ax.plot([-L,L,L,-L,-L],[-L,-L,L,L,-L],'r-',linewidth=2)

def update(frame):
    X=pos[frame,:,0]
    Y=pos[frame,:,1]
    point.set_data(X,Y)
    return point,

ani=FuncAnimation(fig,update,frames=steps,interval=10,blit=True)

plt.show()
ani.save("animationpart.gif",writer='pillow',fps=30)