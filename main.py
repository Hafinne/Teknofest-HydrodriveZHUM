# main.py

import numpy as np
from environment import ThermalEnvironment
from agent import QLearningAgent
from visualizer import (plot_cop_curve, plot_thermal_power,
                         plot_battery_power, plot_energy_consumption,
                         plot_training_rewards, plot_coldstart_comparison)


#  grafiklerini üret

print("Elektrik ekibi verileri görselleştiriliyor...")
plot_cop_curve()
plot_thermal_power()
plot_battery_power()
plot_energy_consumption()

# 2. Q-Learning eğitimi

print("\nQ-Learning eğitimi başlıyor...")

env   = ThermalEnvironment(T_amb=-10.0, SoC_init=0.60)
agent = QLearningAgent(alpha=0.1, gamma=0.95,
                        epsilon=1.0, epsilon_decay=0.995)

N_EPISODES    = 1000
reward_history = []

for episode in range(N_EPISODES):
    state = env.reset(T_amb=-10.0, SoC_init=0.60)
    total_reward = 0

    while True:
        action              = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state        = next_state
        total_reward += reward
        if done:
            break

    agent.decay_epsilon()
    reward_history.append(total_reward)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1:4d} | "
              f"Ödül: {total_reward:8.1f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"T_bat: {info['T_bat']:.1f}°C | "
              f"SoC: {info['SoC']:.2f}")

agent.save("q_table.pkl")
plot_training_rewards(reward_history)


# 3. Test: Q-Learning vs PTC vs Hibrit
print("\nKarşılaştırmalı test başlıyor (-10°C, SoC=0.60)...")

def run_fixed_policy(action_id, T_amb=-10.0, SoC_init=0.60):
    """Sabit stratejiyle test (PTC-only veya Hibrit)"""
    env_test = ThermalEnvironment(T_amb=T_amb, SoC_init=SoC_init)
    env_test.reset()
    total_energy = 0
    steps = 0
    while not env_test.done:
        _, _, done, info = env_test.step(action_id)
        total_energy += info["P_elec"] * env_test.dt / 3600.0
        steps += 1
    return steps * env_test.dt, total_energy

def run_qlearning(agent, T_amb=-10.0, SoC_init=0.60):
    """Eğitilmiş Q-Learning ajanıyla test"""
    env_test = ThermalEnvironment(T_amb=T_amb, SoC_init=SoC_init)
    state = env_test.reset()
    agent.epsilon = 0.0   # test sırasında keşif yok
    total_energy = 0
    steps = 0
    while not env_test.done:
        action = agent.select_action(state)
        state, _, done, info = env_test.step(action)
        total_energy += info["P_elec"] * env_test.dt / 3600.0
        steps += 1
    return steps * env_test.dt, total_energy

t_ql,  e_ql  = run_qlearning(agent)
t_ptc, e_ptc = run_fixed_policy(action_id=1)   # PTC only
t_hyb, e_hyb = run_fixed_policy(action_id=3)   # Hibrit

results = {
    "Q-Learning": {"time": t_ql,  "energy": e_ql},
    "PTC Only":   {"time": t_ptc, "energy": e_ptc},
    "Hybrid":     {"time": t_hyb, "energy": e_hyb},
}

print("\n── Sonuçlar ──")
for method, vals in results.items():
    print(f"{method:12s}: "
          f"Süre={vals['time']:.0f}s, "
          f"Enerji={vals['energy']:.1f} Wh")

plot_coldstart_comparison(results)
print("\nTüm grafikler kaydedildi.")