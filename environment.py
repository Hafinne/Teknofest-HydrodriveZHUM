# environment.py
 
import numpy as np
from data_tables import get_cop, get_qreq, get_p_ptc, get_p_hp, get_p_hybrid
 
class ThermalEnvironment:
 
    def __init__(self, T_amb=-10.0, SoC_init=0.60):
        self.m_battery  = 150
        self.m_fuelcell = 50
        self.c_p        = 1000
        self.T_target   = 20.0
        self.dt         = 10
        self.bat_cap_Wh = 60000
        self.V_bat      = 400
        self.eta_ptc    = 0.96
 
        self.T_amb      = T_amb
        self.T_bat      = T_amb
        self.T_fc       = T_amb
        self.SoC        = SoC_init
        self.n_actions  = 5
        self.done       = False
        self.step_count = 0
        self.max_steps  = 300
 
    def reset(self, T_amb=None, SoC_init=None):
        if T_amb    is not None: self.T_amb = T_amb
        if SoC_init is not None: self.SoC   = SoC_init
        self.T_bat      = self.T_amb
        self.T_fc       = self.T_amb
        self.done       = False
        self.step_count = 0
        return self._discretize_state()
 
    def _discretize_state(self):
        # T_bat: -20 ile +45 arasi, 8 bin
        t_idx   = int(np.clip(int((self.T_bat + 20) / 8), 0, 7))
        # SoC: 0.0 ile 1.0 arasi, 5 bin
        soc_idx = int(np.clip(int(self.SoC / 0.2), 0, 4))
        # T_amb: -20 ile +10 arasi, 6 bin
        ta_idx  = int(np.clip(int((self.T_amb + 20) / 5), 0, 5))
        return (t_idx, soc_idx, ta_idx)
 
    def step(self, action):
        self.step_count += 1
 
        COP   = get_cop(self.T_amb)
        P_ptc = get_p_ptc(self.T_amb)
        P_hp  = get_p_hp(self.T_amb)
 
        if action == 0:
            Q_heat = 0;                    P_elec = 0
        elif action == 1:
            Q_heat = P_ptc * self.eta_ptc; P_elec = P_ptc
        elif action == 2:
            P_elec = P_hp;                 Q_heat = P_hp * COP
        elif action == 3:
            P_elec = get_p_hybrid(self.T_amb)
            Q_heat = P_ptc * self.eta_ptc + P_hp * COP
        elif action == 4:
            Q_heat = -500;                 P_elec = 300
        else:
            Q_heat = 0;                    P_elec = 0
 
        m_total    = self.m_battery + self.m_fuelcell
        dT         = (Q_heat * self.dt) / (m_total * self.c_p)
        self.T_bat = self.T_bat + dT
        self.T_fc  = self.T_fc  + dT * 0.8
 
        E_Wh      = P_elec * self.dt / 3600.0
        self.SoC  = max(0.0, self.SoC - E_Wh / self.bat_cap_Wh)
 
        reward = self._compute_reward(dT, P_elec)
 
        if self.T_bat >= self.T_target: self.done = True
        if self.SoC   <= 0.05:          self.done = True
        if self.step_count >= self.max_steps: self.done = True
 
        info = {
            "T_bat":  self.T_bat,
            "SoC":    self.SoC,
            "P_elec": P_elec,
            "COP":    COP,
            "Q_heat": Q_heat,
        }
        return self._discretize_state(), reward, self.done, info
 
    def _compute_reward(self, dT, P_elec):
        r = 0.0
        r += 3.0 * dT
        r -= 0.0005 * P_elec
        if self.SoC < 0.2:      r -= 30
        if self.T_bat > 45:     r -= 80
        if self.T_bat >= self.T_target: r += 300
        return r