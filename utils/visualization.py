import numpy as np
from pathlib import Path
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt; _MPL=True
except: _MPL=False
try:
    import pygame; _PG=True
except: _PG=False

C = {
    'bg':(18,18,26),'wall':(15,15,22),'aisle':(235,235,228),
    'rack':(75,65,50),'rack_req':(210,140,20),'goal':(25,75,170),
    'charge':(235,195,10),'panel':(20,20,30),'txt':(215,215,215),
    'tdim':(120,120,140),
    'agv':  [(35,115,235),(235,95,25),(25,190,90),(220,70,70),(120,110,255),(40,170,160)],
    'pick': [(195,45,195),(45,195,195),(255,160,60),(180,220,90)],
}
MPL_AGENT_COLORS = ['#4f8cff', '#ff8c42', '#43c985', '#d66bff', '#00b7c2', '#e85d75', '#a0c25b', '#7a88ff']

def _hex(surf,clr,cx,cy,r,bclr=(255,255,255),bw=2):
    pts=[(cx+r*np.cos(np.radians(60*i-30)),
          cy+r*np.sin(np.radians(60*i-30))) for i in range(6)]
    pygame.draw.polygon(surf,clr,pts)
    if bw: pygame.draw.polygon(surf,bclr,pts,bw)

def _diamond(surf,clr,cx,cy,r,bclr=(255,255,255),bw=2):
    pts=[(cx,cy-r),(cx+r,cy),(cx,cy+r),(cx-r,cy)]
    pygame.draw.polygon(surf,clr,pts)
    if bw: pygame.draw.polygon(surf,bclr,pts,bw)


class Renderer:
    CELL=32
    def __init__(self,env):
        if not _PG: self._ok=False; return
        self._ok=True
        self.env=env
        CL=self.CELL
        panel_h = max(170, 92 + env.n_agents * 18)
        self.W=env.W*CL; self.H2=env.H*CL + panel_h
        self.panel_h = panel_h
        pygame.init()
        self.scr=pygame.display.set_mode((self.W,self.H2))
        pygame.display.set_caption("TA-RWARE Pro v2")
        self.clk=pygame.time.Clock()
        self.fn=pygame.font.SysFont('consolas',15)
        self.fs=pygame.font.SysFont('consolas',12)
        self.fb=pygame.font.SysFont('consolas',18,bold=True)

    def render(self,env,info,ep=0):
        if not self._ok: return True
        import pygame as pg
        for ev in pg.event.get():
            if ev.type==pg.QUIT: return False
        CL=self.CELL
        scr=self.scr
        scr.fill(C['bg'])

        from envs.warehouse_env import Cell as Ct, AgentType, Phase
        for r in range(env.H):
            for c in range(env.W):
                cv=env.grid[r,c]
                rect=pg.Rect(c*CL,r*CL,CL-1,CL-1)
                clr=(C['wall'] if cv==Ct.WALL else
                     C['rack'] if cv==Ct.RACK else
                     C['goal'] if cv==Ct.GOAL else
                     C['charge'] if cv==Ct.CHARGE else C['aisle'])
                pg.draw.rect(scr,clr,rect)

        # Highlight requested racks
        req_pos={tuple(o.rack_pos) for o in env.orders if not o.done}
        for (rc,rr) in req_pos:
            pg.draw.rect(scr,C['rack_req'],
                         pg.Rect(rc*CL+2,rr*CL+2,CL-4,CL-4))
            scr.blit(self.fs.render("!",True,(255,255,255)),
                     (rc*CL+CL//2-4,rr*CL+CL//2-6))

        # Order lines
        for o in env.orders:
            if o.done or o.agv_id<0: continue
            agv=next((a for a in env.agents if a.id==o.agv_id),None)
            if not agv: continue
            sx=agv.pos[0]*CL+CL//2; sy=agv.pos[1]*CL+CL//2
            if agv.phase in (Phase.TO_PICKUP,):
                tgt=o.rack_pos
            else:
                tgt=o.goal_pos or o.rack_pos
            if tgt:
                pg.draw.line(scr,(100,200,100,80),(sx,sy),
                             (tgt[0]*CL+CL//2,tgt[1]*CL+CL//2),1)

        # Agents
        for a in env.agents:
            cx=a.pos[0]*CL+CL//2; cy=a.pos[1]*CL+CL//2; r=CL//2-3
            if a.type==AgentType.AGV:
                base=C['agv'][a.id%len(C['agv'])]
                fill=tuple(min(255,x+60) for x in base) if a.phase == Phase.TO_GOAL else base
                _hex(scr,fill,cx,cy,r)
            else:
                pi=a.id-env.n_agvs
                base=C['pick'][pi%len(C['pick'])]
                _diamond(scr,base,cx,cy,r)
            # Battery bar
            bp=a.battery/a.batt_cap
            bc=(60,220,60) if bp>0.5 else (220,200,0) if bp>0.2 else (220,60,60)
            bl=int((CL-6)*bp)
            pg.draw.rect(scr,(40,40,40),pg.Rect(cx-CL//2+3,cy+r+2,CL-6,3))
            pg.draw.rect(scr,bc,pg.Rect(cx-CL//2+3,cy+r+2,bl,3))
            scr.blit(self.fs.render(str(a.id),True,(255,255,255)),(cx-4,cy-6))
            if a.charging:
                scr.blit(self.fs.render("C",True,C['charge']),(cx+r-2,cy-r))

        # Panel
        py=env.H*CL
        pg.draw.rect(scr,C['panel'],pg.Rect(0,py,self.W,self.panel_h))
        lines=[
            (f"Episode {ep}   Step {info['steps']}/{env.max_steps}",self.fb),
            (f"Deliveries: {info['deliveries']} / {info['total_orders']}   Pending: {info['pending_orders']}",self.fn),
            (f"Total Reward: {info['total_reward']:+.2f}",self.fn),
        ]
        for li,(t,fn) in enumerate(lines):
            scr.blit(fn.render(t,True,C['txt']),(8,py+4+li*22))
        # Per-agent
        for i,am in enumerate(info['agent_metrics']):
            atype="AGV   " if i<env.n_agvs else "Picker"
            t=f"Agent{i}({atype}): del={am['deliveries']} dist={am['distance']} coll={am['collisions']}"
            base=C['agv'][i%len(C['agv'])] if i<env.n_agvs else C['pick'][(i-env.n_agvs)%len(C['pick'])]
            scr.blit(self.fs.render(t,True,base),(8,py+72+i*16))
        # Legend
        lx=self.W-160
        for li,(txt,clr) in enumerate([("Hex=AGV",C['agv'][0]),
                                        ("Diamond=Picker",C['pick'][0]),
                                        ("Yellow rack=Order",C['rack_req']),
                                        ("Blue=Goal",C['goal']),
                                        ("Yellow=Charger",C['charge'])]):
            pg.draw.rect(scr,clr,pg.Rect(lx,py+6+li*18,10,10))
            scr.blit(self.fs.render(txt,True,C['tdim']),(lx+14,py+4+li*18))
        pg.display.flip()
        self.clk.tick(6)
        return True

    def close(self):
        if self._ok: pygame.quit()


def plot_metrics(metrics, path):
    if not _MPL: return
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    fig.patch.set_facecolor('#12121a')
    for ax in axes.flat:
        ax.set_facecolor('#1e1e2e')
        ax.tick_params(colors='#aaaacc')
        ax.spines[:].set_color('#333355')

    def _p(ax,data,title,color,ylabel=''):
        if not data: return
        x=np.arange(len(data))
        ax.plot(x,data,color=color,alpha=0.3,lw=0.8)
        w=max(1,len(data)//50)
        sm=np.convolve(data,np.ones(w)/w,mode='valid')
        ax.plot(np.arange(len(sm)),sm,color=color,lw=2)
        ax.set_title(title,color='#ccccee',fontsize=11)
        ax.set_ylabel(ylabel,color='#aaaacc',fontsize=9)
        ax.set_xlabel('Episode',color='#aaaacc',fontsize=9)
        ax.grid(True,alpha=0.2)

    _p(axes[0,0],metrics.get('episode_rewards',[]),  'Episode Reward',    '#4488ff','Reward')
    _p(axes[0,1],metrics.get('deliveries',[]),        'Deliveries/Ep',     '#44cc88','Count')
    _p(axes[0,2],metrics.get('completion_rates',[]),  'Completion %',      '#ffaa44','%')
    _p(axes[1,0],metrics.get('losses',[]),            'Training Loss',     '#ff6666','Loss')
    _p(axes[1,1],metrics.get('q_values',[]),          'Mean Q-Value',      '#66aaff','Q')
    _p(axes[1,2],metrics.get('epsilons',[]),          'Epsilon',           '#aa66ff','eps')
    plt.tight_layout(pad=2)
    plt.savefig(path,dpi=120,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved -> {path}")

    agent_metrics = metrics.get('agent_episode_metrics', {})
    agent_rewards = agent_metrics.get('reward', {})
    if not agent_rewards:
        return

    def _agent_plot(ax, series, title, ylabel):
        ax.set_facecolor('#1e1e2e')
        ax.tick_params(colors='#aaaacc')
        ax.spines[:].set_color('#333355')
        ax.set_title(title, color='#ccccee', fontsize=11)
        ax.set_ylabel(ylabel, color='#aaaacc', fontsize=9)
        ax.set_xlabel('Episode', color='#aaaacc', fontsize=9)
        ax.grid(True, alpha=0.2)
        if not series:
            return
        for idx, (label, values) in enumerate(series.items()):
            if not values:
                continue
            x = np.arange(len(values))
            w = max(1, len(values)//50)
            sm = np.convolve(values, np.ones(w)/w, mode='valid')
            ax.plot(np.arange(len(sm)), sm,
                    color=MPL_AGENT_COLORS[idx % len(MPL_AGENT_COLORS)],
                    lw=2, label=label)
        ax.legend(fontsize=8)

    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
    fig2.patch.set_facecolor('#12121a')
    _agent_plot(axes2[0,0], agent_metrics.get('reward', {}),     'Agent Reward Comparison',     'Reward')
    _agent_plot(axes2[0,1], agent_metrics.get('deliveries', {}), 'Agent Deliveries Comparison', 'Deliveries')
    _agent_plot(axes2[0,2], agent_metrics.get('assists', {}),    'Agent Assists Comparison',    'Assists')
    _agent_plot(axes2[1,0], agent_metrics.get('distance', {}),   'Agent Distance Comparison',   'Distance')
    _agent_plot(axes2[1,1], agent_metrics.get('collisions', {}), 'Agent Collision Comparison',  'Collisions')

    ax_bar = axes2[1,2]
    ax_bar.set_facecolor('#1e1e2e')
    ax_bar.tick_params(colors='#aaaacc')
    ax_bar.spines[:].set_color('#333355')
    ax_bar.set_title('Recent Avg Reward (last 100 eps)', color='#ccccee', fontsize=11)
    labels, means = [], []
    for label, values in agent_rewards.items():
        if values:
            labels.append(label)
            means.append(float(np.mean(values[-100:])))
    ax_bar.bar(labels, means, color=MPL_AGENT_COLORS[:len(labels)])
    ax_bar.tick_params(axis='x', rotation=20)
    ax_bar.set_ylabel('Reward', color='#aaaacc', fontsize=9)
    ax_bar.grid(True, axis='y', alpha=0.2)

    plt.tight_layout(pad=2)
    p = Path(path)
    agent_path = p.with_name(f"{p.stem}_agents{p.suffix}")
    plt.savefig(agent_path, dpi=120, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Plot saved -> {agent_path}")
