
# 🧠 Slime Volleyball AI Agents: Minimax & Alpha-Beta Pruning

This project implements **Minimax** and **Alpha-Beta Pruning** algorithms on the classic **Slime Volleyball** environment using Python and OpenAI Gym. It also includes a **Random Agent** for baseline comparison and scripts to simulate, evaluate, and record gameplay.

---

## 📁 Repository Structure

```
AI_3/
├── algorithm/
│   ├── alphabeta.py          # Alpha-Beta pruning implementation
│   ├── minimax.py            # Minimax algorithm implementation
│   └── random_agent.py       # Random agent for baseline testing
│
├── slimevolleygym/
│   ├──__init__.py
│   ├── mlp.py            # MLP (used internally by environment agents)
│   ├── patch_env.py      # Gym environment patch
│   ├── slimevolley.py    # Main environment logic
│   └── __pycache__/      # Python cache files
│
├── minimax_fullgame_0.5x1.mp4     # Output video (Minimax)
├── alphabeta_fullgame_0.5x1.mp4   # Output video (Alpha-Beta)
│
├── final.py                # Script to run both algorithms and record matches
├── basic_test.py           # Test script to verify environment setup
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


---

## ⚙️ Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Ashantfet/AI_3.git
   cd AI_3
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Run both agents and save full gameplay:
```bash
python final.py
```

### Run Minimax agent only:
```bash
python eval_minimax.py
```

### Run Alpha-Beta agent only:
```bash
python eval_alphabeta.py
```

Each script records a full **30-second match at 30 fps ** and saves the video for visualization.

---

## 🧪 Evaluation

| Agent         | Score (Yellow vs. Blue) | Frames | Execution Time |
|---------------|--------------------------|--------|----------------|
| Minimax       | 0 - 3                    | 900    | 227.23 s       |
| Alpha-Beta    | 0 - 13                    | 900    | 109.03 s       |

- **Blue** was the **maximizing player**.
- **Alpha-Beta Pruning** provided identical performance to Minimax but with nearly **2x faster execution**.

---

## 🎯 Evaluation Function

Both algorithms used a **simple evaluation function** based on:
- Distance between the agent and the ball.
- Positional advantage on the field.

This heuristic allowed reasonable control and decision-making for the agents.

---

## 🔍 Key Insights

- **Minimax vs. Alpha-Beta:** Same results, Alpha-Beta is faster.
- **Random Agent:** No strategy, poor performance.
- **Evaluation:** Effective, but limited by simplistic features.

---

## 🚧 Challenges & Future Work

- **Challenges:**
  - Limited planning depth affects long-term strategies.
  - Difficulty modeling dynamic opponent and ball physics.

- **Future Work:**
  - Improve evaluation using ball velocity, opponent position, etc.
  - Integrate **reinforcement learning** for adaptive strategy learning.

---

## 📽️ Demo Videos

- [🎮 Minimax Gameplay](minimax_fullgame.mp4)
- [🎮 Alpha-Beta Gameplay](alphabeta_fullgame.mp4)

---
---
## 📽️ Presentation

- [Google Slides Link](https://docs.google.com/presentation/d/1YvWjGo5Mkzojhy9obGW0f-NMrgvtPCzGR4rOFVKiPoE/edit?usp=sharing)
## 🤝 Credits

Developed as part of **AI Assignment 3** @ IIT TIRUPATI
Authors: ASHANT KUMAR (CS24M113)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
```

---

Let me know if you want the README tailored with your GitHub URL, author name, or institution!
