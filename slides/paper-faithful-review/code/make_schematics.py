"""Explanatory figures for the NBD deck; no benchmark models are trained here.

All point clouds are schematic. The diffusion illustration uses one small,
explicitly constructed graph. Each output is its own Matplotlib figure.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
MANIFEST = {}


def save(fig, name, description):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight', pad_inches=0.055)
    MANIFEST[name] = {'kind': 'schematic', 'description': description}
    plt.close(fig)


def blank(size=(6.4, 3.8), xlim=(-2.6, 2.6), ylim=(-1.5, 1.6)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal'); ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return fig, ax


def arrow(ax, p, q, **kwargs):
    opts = {'arrowstyle': '->', 'lw': 1.8, 'mutation_scale': 15}
    opts.update(kwargs)
    ax.annotate('', xy=q, xytext=p, arrowprops=opts)


def linecurve(t):
    return .42 * np.sin(1.35*t) + .12*t


def ell(ax, c, width=1.15, height=.32, angle=0, **kwargs):
    e = Ellipse(c, width, height, angle=angle, fill=False, lw=1.65, **kwargs)
    ax.add_patch(e)
    return e


# 1. Normal-only training and mixed test points.
fig, ax = blank(xlim=(-2.3, 2.35), ylim=(-1.15, 1.75))
t = np.linspace(-1.95, 1.95, 120)
ax.plot(t, linecurve(t), '--', lw=1.8, alpha=.6)
u = RNG.uniform(-1.95, 1.95, 75)
ax.scatter(u, linecurve(u)+RNG.normal(0,.09,len(u)), s=28, alpha=.7, label='Normal')
a = np.array([[-1.1,1.18],[.1,1.18],[1.25,-.65],[1.75,1.4]])
ax.scatter(a[:,0], a[:,1], marker='x', s=95, lw=2.0, label='Anomaly')
ax.legend(loc='lower left', ncol=2, frameon=False, fontsize=11)
save(fig, 'problem', 'Normal samples lie near a curved support; test anomalies need not.')

# 2. Nearest-memory score.
fig, ax = blank(xlim=(-2.4,2.4), ylim=(-1.3,1.7))
u = RNG.uniform(-2,2,72)
ax.scatter(u, linecurve(u)+RNG.normal(0,.07,len(u)), s=18, alpha=.25)
mt = np.linspace(-1.9,1.9,7)
mem = np.column_stack([mt,linecurve(mt)])
ax.scatter(mem[:,0],mem[:,1],s=55,marker='s',label='Memory vectors')
q = np.array([.65,1.37])
ax.scatter(*q,marker='x',s=110,lw=2.2,label='Query')
j = np.argmin(np.sum((mem-q)**2,axis=1))
arrow(ax,mem[j],q,linestyle='--')
ax.text(q[0]+.14,q[1],r'$z$',fontsize=17)
ax.text(mem[j,0]+.10,mem[j,1]-.34,r'$m_{j^*}$',fontsize=16)
ax.text(-2.1,1.30,r'$j^*=\arg\min_j\,\|z-m_j\|^2$',fontsize=14)
ax.legend(loc='lower left',frameon=False,ncol=2,fontsize=10)
save(fig, 'patchscore', 'Query scored by squared distance to the nearest stored normal vector.')

# 3. SVDD: a sphere in a fixed kernel feature space.
fig, ax = blank(size=(5.6,2.75),xlim=(-2.0,2.0),ylim=(-1.16,1.16))
c=np.array([-.18,0.0]); rad=.92
ax.add_patch(Circle(c,rad,fill=False,lw=1.8))
p=RNG.normal(size=(38,2)); p=p/np.maximum(1,np.linalg.norm(p,axis=1,keepdims=True))*RNG.uniform(.2,.82,(38,1))+c
ax.scatter(p[:,0],p[:,1],s=19,alpha=.6)
ax.scatter(*c,marker='+',s=90,lw=2)
q=c+np.array([1.45,.55]);ax.scatter(*q,marker='x',s=75,lw=2)
arrow(ax,c,c+np.array([.69,-.61]))
ax.text(.28,-.38,r'$R$',fontsize=14)
ax.text(c[0]-.24,c[1]+.04,r'$a$',fontsize=14)
ax.text(q[0]+.08,q[1]+.05,r'$\phi_k(x)$',fontsize=14)
save(fig,'svdd','Illustrative enclosing circle in fixed kernel feature space; not a fitted decision boundary.')

# 4. Deep SVDD: learn a map toward a fixed center.
fig, ax=blank(size=(5.6,2.75),xlim=(-2.6,2.6),ylim=(-1.16,1.16))
p=RNG.normal(size=(32,2))*[.4,.32]+[-1.65,0]
ax.scatter(p[:,0],p[:,1],s=19,alpha=.60)
q=RNG.normal(size=(32,2))*[.20,.20]+[1.0,0]
ax.scatter(q[:,0],q[:,1],s=19,alpha=.7)
arrow(ax,[-.73,0],[.43,0]);ax.text(-.25,.21,r'$f_\theta$',fontsize=15,ha='center')
ax.scatter(1,0,marker='+',s=100,lw=2)
ax.text(.92,-.38,r'$c$',fontsize=14)
ax.text(-1.65,-.91,'Input',fontsize=11,ha='center')
ax.text(1.1,-.91,'Learned features',fontsize=11,ha='center')
save(fig,'deep_svdd','Schematic learned mapping concentrating normal representations near a fixed center.')

# 5. DROCC's annular local negative search.
fig, ax=blank(size=(5.6,3.3),xlim=(-1.65,1.8),ylim=(-1.37,1.38))
t=np.linspace(-1.4,1.4,45)
ax.scatter(t,.06*np.sin(t*2),s=14,alpha=.38)
ax.add_patch(Circle((0,0),.52,fill=False,ls=':',lw=1.8))
ax.add_patch(Circle((0,0),1.04,fill=False,ls='--',lw=1.8))
ax.scatter(0,0,s=70,marker='o')
ps=np.array([[.59,.63],[-.62,.51],[-.3,-.81]])
ax.scatter(ps[:,0],ps[:,1],s=80,marker='x',lw=2)
arrow(ax,[0,0],ps[0]);ax.text(.19,.39,r'$h$',fontsize=15)
ax.text(.7,.67,r'$x_i+h$',fontsize=14)
ax.text(-.17,-.25,r'$x_i$',fontsize=14)
arrow(ax,[0,0],[.5,-.04],lw=1.1)
ax.text(.35,-.22,r'$r$',fontsize=14)
ax.text(-1.19,-.63,r'$\gamma r$',fontsize=14)
ax.text(-1.63,1.15,'Search in the annulus',fontsize=12)
save(fig,'drocc','Schematic pseudo-negative search around a normal point; the annulus does not exclude all other normals.')

# 6. A curved two-dimensional surface embedded in three dimensions.
fig=plt.figure(figsize=(8.0,4.5))
ax=fig.add_subplot(111,projection='3d')
def surf(x,y): return .28*np.sin(1.25*x)+.13*x*y+.14*np.cos(1.4*y)
xg,yg=np.meshgrid(np.linspace(-1.9,1.9,24),np.linspace(-1.35,1.35,19))
zg=surf(xg,yg)
ax.plot_wireframe(xg,yg,zg,rstride=2,cstride=2,lw=.7,alpha=.23)
uv=RNG.uniform([-1.85,-1.25],[1.85,1.25],(90,2))
ax.scatter(uv[:,0],uv[:,1],surf(uv[:,0],uv[:,1])+RNG.normal(0,.025,len(uv)),s=7,alpha=.23)
for j,(x,y) in enumerate([(-1.28,-.68),(-.30,-.55),(.8,-.35),(-.88,.62),(.38,.66),(1.35,.67)],1):
    c=np.array([x,y,surf(x,y)])
    ux=np.array([1,0,.35*np.cos(1.25*x)+.13*y]);ux/=np.linalg.norm(ux)
    vy=np.array([0,1,.13*x-.196*np.sin(1.4*y)]);vy-=ux*np.dot(ux,vy);vy/=np.linalg.norm(vy)
    n=np.cross(ux,vy)
    az,po=np.meshgrid(np.linspace(0,2*np.pi,25),np.linspace(0,np.pi,13))
    xyz=(c[:,None,None]+.39*ux[:,None,None]*np.sin(po)*np.cos(az)
         +.30*vy[:,None,None]*np.sin(po)*np.sin(az)+.065*n[:,None,None]*np.cos(po))
    ax.plot_wireframe(*xyz,rstride=3,cstride=4,lw=.9,alpha=.9)
    ax.scatter(*c,s=21)
    if j in [1,3,6]: ax.text(*(c+[0,0,.18]),rf'$B_{j}$',fontsize=13)
ax.set_xlim(-2.05,2.05);ax.set_ylim(-1.5,1.5);ax.set_zlim(-.65,.92)
ax.set_box_aspect((4.1,3,1.57));ax.view_init(elev=26,azim=-61);ax.set_axis_off()
fig.subplots_adjust(0,0,1,1)
save(fig,'manifold3d','Analytic curved surface with tangent-aligned ellipsoids; a schematic, not benchmark data.')

# 7. Exact local tangent / residual decomposition in R².
fig,ax=blank(size=(6.1,3.5),xlim=(-1.9,2.0),ylim=(-1.0,1.68))
th=np.deg2rad(23);u=np.array([np.cos(th),np.sin(th)]);n=np.array([-u[1],u[0]])
ell(ax,[0,0],width=2.9,height=.55,angle=23)
ax.plot([-1.7*u[0],1.7*u[0]],[-1.7*u[1],1.7*u[1]],ls='--',lw=1.1,alpha=.65)
foot=.85*u; z=foot+.78*n
ax.scatter(0,0,s=45);ax.scatter(*z,s=90,marker='x',lw=2)
arrow(ax,[0,0],foot,lw=2);arrow(ax,foot,z,lw=2,linestyle='--')
ax.plot([foot[0]],[foot[1]],marker='o',ms=4)
ax.text(-.30,-.28,r'$c_j$',fontsize=15)
ax.text(z[0]+.1,z[1]+.09,r'$z$',fontsize=17)
ax.text(.7,-.15,r'$U_j a_j$',fontsize=16)
ax.text(1.0,.90,r'$r_{\perp,j}$',fontsize=16)
arrow(ax,[.98,.93],(foot+z)/2,lw=.8)
ax.text(-1.7,1.39,'Tangent motion + normal residual',fontsize=12)
save(fig,'bubble_detail','A query projected onto the local tangent line; the dashed leg is the perpendicular residual.')

# 8. Three pairwise terms, shown separately so each figure can stay legible.
fig,ax=blank(size=(4.7,2.5),xlim=(-1.65,1.65),ylim=(-.8,.95))
i=np.array([-.85,-.12]);j=np.array([.87,.2])
ell(ax,i,width=1.0,height=.33,angle=13);ell(ax,j,width=1.0,height=.33,angle=13)
ax.scatter([i[0],j[0]],[i[1],j[1]],s=40)
arrow(ax,i,j,linestyle='--',lw=1.4)
ax.text(-.91,-.62,r'$c_i$',fontsize=15);ax.text(.85,-.39,r'$c_j$',fontsize=15)
ax.text(-.24,.45,r'$c_j-c_i$',fontsize=14)
save(fig,'pair_centers','Displacement between the two bubble centers.')

fig,ax=blank(size=(4.7,2.5),xlim=(-1.65,1.65),ylim=(-.8,.95))
ell(ax,[0,0],width=2.0,height=.29,angle=23)
ell(ax,[0,0],width=2.0,height=.29,angle=-23,ls='--')
for th in [23,-23]:
 d=np.array([np.cos(np.deg2rad(th)),np.sin(np.deg2rad(th))]);arrow(ax,[0,0],1.27*d,lw=1.3)
ax.text(1.19,.56,r'$\Pi_i$',fontsize=16);ax.text(1.19,-.52,r'$\Pi_j$',fontsize=16)
ax.scatter(0,0,s=22)
save(fig,'pair_frames','Projectors compare subspaces, independently of sign changes in a tangent basis.')

fig,ax=blank(size=(4.7,2.5),xlim=(-1.65,1.65),ylim=(-.8,.95))
ell(ax,[-.80,.10],width=1.22,height=.22)
ell(ax,[.80,.10],width=1.22,height=.64)
ax.scatter([-.80,.8],[.1,.1],s=26)
ax.text(-.81,-.53,r'$\sigma_{\perp,i}$',fontsize=16,ha='center')
ax.text(.81,-.53,r'$\sigma_{\perp,j}$',fontsize=16,ha='center')
save(fig,'pair_thickness','Optional comparison of residual thickness; not a full covariance distance.')

# 9. One graph shared by the graph and diffusion illustrations.
t=np.linspace(-2.7,2.7,11)
C=np.c_[t,.50*np.sin(1.15*t)]
U=np.c_[np.ones(len(t)),.575*np.cos(1.15*t)]
U/=np.linalg.norm(U,axis=1,keepdims=True)
G=np.array([np.outer(u,u)/.75**2+(np.eye(2)-np.outer(u,u))/.30**2 for u in U])
N=len(t); W=np.eye(N)*.6
for i in range(N):
 for j in range(i+1,min(N,i+3)):
  delta=C[i]-C[j]
  rc=.5*delta@(G[i]+G[j])@delta
  rf=np.linalg.norm(np.outer(U[i],U[i])-np.outer(U[j],U[j]))**2
  W[i,j]=W[j,i]=np.exp(-rc/3-.45*rf)
s=W.sum(axis=1);P=W/s[:,None];pi=s/s.sum()
assert np.allclose(P.sum(axis=1),1) and np.allclose(pi@P,pi)
fig,ax=blank(size=(7,3.1),xlim=(-3.15,3.15),ylim=(-1.0,1.0))
segments=[]; widths=[]
for i in range(N):
 for j in range(i+1,N):
  if W[i,j]>0:
   segments.append(C[[i,j]]); widths.append(.5+2.8*W[i,j])
ax.add_collection(LineCollection(segments,linewidths=widths,alpha=.45))
ax.scatter(C[:,0],C[:,1],s=115,zorder=3)
for j in [0,4,7,10]:
 ax.annotate(rf'$B_{{{j+1}}}$',C[j],xytext=(0,20 if j>5 else -23),textcoords='offset points',fontsize=14,ha='center')
save(fig,'bubble_graph','Toy graph with bubble nodes and symmetric local edge weights; line width encodes affinity.')

# 10. Distribution view of diffusion on exactly the same toy graph.
Pt=np.linalg.matrix_power(P,3)
fig,ax=plt.subplots(figsize=(6.4,3.1),layout='constrained')
ax.plot(np.arange(1,N+1),Pt[4],marker='o',lw=2,label=r'Start at $B_5$')
ax.plot(np.arange(1,N+1),Pt[5],marker='s',ls='--',lw=2,label=r'Start at $B_6$')
ax.set_xlabel('Bubble node',fontsize=12);ax.set_ylabel('3-step probability',fontsize=12)
ax.set_xticks([1,3,5,7,9,11]);ax.set_ylim(0,None)
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,fontsize=10,loc='upper right')
save(fig,'diffusion_walks','Two three-step transition distributions on the toy bubble graph, not a benchmark result.')
MANIFEST['diffusion_walks']['kind']='computed toy illustration'

# 11. Query dispersion in diffusion-coordinate space, two separate figures.
coords=np.array([[-1.4,-.20],[-.98,.32],[-.65,-.29],[.88,-.16],[1.36,.25],[1.58,-.37]])
for name,ids,weights in [('attachment_near',[0,1,2],[.25,.5,.25]),('attachment_far',[0,1,4],[.25,.25,.5])]:
 fig,ax=blank(size=(5.3,2.4),xlim=(-1.98,2.0),ylim=(-.78,1.0))
 ax.scatter(coords[:,0],coords[:,1],s=42,alpha=.30)
 pos=coords[ids];b=np.asarray(weights);mean=(pos*b[:,None]).sum(axis=0)
 for p,w in zip(pos,b):
  ax.plot([p[0],mean[0]],[p[1],mean[1]],ls='--',lw=1.3)
 ax.scatter(pos[:,0],pos[:,1],s=420*b,marker='o')
 ax.scatter(*mean,marker='*',s=165)
 ax.text(mean[0]-.04,mean[1]+.53,r'$\bar\Phi_t(z)$',fontsize=14,ha='center')
 save(fig,name,'Illustrative attachment mass in diffusion coordinates; circle area is b_j and star is the weighted mean.')

# Useful reproducibility checks for the computed toy graph.
D2=np.sum((Pt[:,None,:]-Pt[None,:,:])**2/pi[None,None,:],axis=2)
assert np.allclose(D2,D2.T) and np.allclose(np.diag(D2),0)
(OUT/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
np.savez(OUT/'toy_graph.npz',centers=C,W=W,P=P,pi=pi,D3_squared=D2)
print(f'{len(MANIFEST)} illustrations saved as PDF and PNG. No benchmark rerun.')

# Remove the excess whitespace reserved by Matplotlib's 3-D axes, retaining vector PDF.
from PIL import Image
import fitz
png=OUT/'manifold3d.png'
im=Image.open(png).convert('RGB')
pixels=np.asarray(im)
ys,xs=np.where(np.any(pixels<245,axis=2))
pad=20
box=(max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),
     min(im.width,int(xs.max())+pad+1),min(im.height,int(ys.max())+pad+1))
pdf=OUT/'manifold3d.pdf'
with fitz.open(pdf) as doc:
    page=doc[0]; w,h=page.rect.width,page.rect.height
    page.set_cropbox(fitz.Rect(box[0]/im.width*w,box[1]/im.height*h,
                              box[2]/im.width*w,box[3]/im.height*h))
    doc.save(OUT/'manifold3d_cropped.pdf')
(OUT/'manifold3d_cropped.pdf').replace(pdf)
im.crop(box).save(png)
