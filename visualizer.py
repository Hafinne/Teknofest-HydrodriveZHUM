# visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from data_tables import (T_AMB_LIST, COP_LIST, QREQ_LIST,
                          P_PTC_LIST, P_HP_LIST, P_HYBRID_LIST,
                          E_PTC_LIST, E_HP_LIST, E_HYBRID_LIST)

def plot_cop_curve():
    """Grafik 1: Elektrik raporuyla aynı COP eğrisi"""
    plt.figure(figsize=(8, 5))
    plt.plot(T_AMB_LIST, COP_LIST, 'b-', linewidth=2)
    plt.xlabel("Ortam Sıcaklığı (°C)")
    plt.ylabel("Isı Pompası COP")
    plt.title("Ambient Temperature vs Heat Pump COP")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig1_cop.png", dpi=150)
    plt.show()

def plot_thermal_power():
    """Grafik 2: Gerekli ısıl güç"""
    plt.figure(figsize=(8, 5))
    plt.plot(T_AMB_LIST, QREQ_LIST, 'b-', linewidth=2)
    plt.xlabel("Ortam Sıcaklığı (°C)")
    plt.ylabel("Required Thermal Power (W)")
    plt.title("Ambient Temperature vs Required Thermal Power")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig2_qreq.png", dpi=150)
    plt.show()

def plot_battery_power():
    """Grafik 3: Batarya güç tüketimi karşılaştırması"""
    plt.figure(figsize=(8, 5))
    plt.plot(T_AMB_LIST, P_PTC_LIST,    'b-', label="PTC Only",       linewidth=2)
    plt.plot(T_AMB_LIST, P_HP_LIST,     'r-', label="Heat Pump Only", linewidth=2)
    plt.plot(T_AMB_LIST, P_HYBRID_LIST, 'y-', label="Hybrid",         linewidth=2)
    plt.xlabel("Ortam Sıcaklığı (°C)")
    plt.ylabel("Battery Power Demand (W)")
    plt.title("Battery Power Demand vs Ambient Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig3_power.png", dpi=150)
    plt.show()

def plot_energy_consumption():
    """Grafik 5: 5 dakika enerji tüketimi"""
    plt.figure(figsize=(8, 5))
    plt.plot(T_AMB_LIST, E_PTC_LIST,    'b-', label="PTC Only",       linewidth=2)
    plt.plot(T_AMB_LIST, E_HP_LIST,     'r-', label="Heat Pump Only", linewidth=2)
    plt.plot(T_AMB_LIST, E_HYBRID_LIST, 'y-', label="Hybrid",         linewidth=2)
    plt.xlabel("Ortam Sıcaklığı (°C)")
    plt.ylabel("Energy Consumption in 5 min (Wh)")
    plt.title("Energy Consumption vs Ambient Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig5_energy.png", dpi=150)
    plt.show()

def plot_training_rewards(reward_history):
    """Q-Learning eğitim ödül grafiği"""
    plt.figure(figsize=(10, 4))
    plt.plot(reward_history, alpha=0.4, color='gray', label="Ham ödül")
    # Hareketli ortalama
    window = 50
    if len(reward_history) >= window:
        avg = np.convolve(reward_history,
                          np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(reward_history)), avg,
                 'r-', linewidth=2, label=f"{window}-episode ortalama")
    plt.xlabel("Episode")
    plt.ylabel("Toplam Ödül")
    plt.title("Q-Learning Eğitim Performansı")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_training.png", dpi=150)
    plt.show()

def plot_coldstart_comparison(results_dict):
    """
    Q-Learning vs PTC-only vs Hibrit soğuk başlatma karşılaştırması
    results_dict = {
        "Q-Learning": {"time": X, "energy": Y},
        "PTC Only":   {"time": X, "energy": Y},
        "Hybrid":     {"time": X, "energy": Y},
    }
    """
    methods = list(results_dict.keys())
    times   = [results_dict[m]["time"]   for m in methods]
    energies= [results_dict[m]["energy"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#2196F3', '#FF5722', '#FFC107']
    ax1.bar(methods, times, color=colors)
    ax1.set_ylabel("Soğuk Başlatma Süresi (saniye)")
    ax1.set_title("Hedef Sıcaklığa Ulaşma Süresi")
    ax1.grid(axis='y')

    ax2.bar(methods, energies, color=colors)
    ax2.set_ylabel("Enerji Tüketimi (Wh)")
    ax2.set_title("Toplam Enerji Tüketimi")
    ax2.grid(axis='y')

    plt.tight_layout()
    plt.savefig("fig_comparison.png", dpi=150)
    plt.show()