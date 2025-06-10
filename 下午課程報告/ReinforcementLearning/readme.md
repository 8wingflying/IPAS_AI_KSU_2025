# 強化學習| 增強式學習 |Reinforcement Learning(RL)報告(2025/10)
- 強化學習
- 強化學習架構
  - 簡單示意圖
  - 馬爾可夫決策過程（Markov decision processes，MDP） 
- 強化學習演算法
- 強化學習範例學習


# YOUTUBE教學影片
### [李弘毅教授| Hung-yi Lee](https://www.youtube.com/@HungyiLeeNTU)
- [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (一) – 增強式學習跟機器學習一樣都是三個步驟](https://www.youtube.com/watch?v=XWukX-ayIrs&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=30)
- [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (二) – Policy Gradient 與修課心情](https://www.youtube.com/watch?v=US8DFaAZcp4&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=31)
- [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (三) - Actor-Critic](https://www.youtube.com/watch?v=kk6DqWreLeU&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=32)
- [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (四) - 回饋非常罕見的時候怎麼辦？機器的望梅止渴](https://www.youtube.com/watch?v=73YyF1gmIus&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=33)
- [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL) (五) - 如何從示範中學習？逆向增強式學習 (Inverse RL)](https://www.youtube.com/watch?v=75rZwxKBAf0&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=34)
- https://www.youtube.com/watch?v=75rZwxKBAf0&t=9s
- https://sumniya.tistory.com/2

## 延伸閱讀
- [Deep Reinforcement Learning Hands-On - Third Edition](https://learning.oreilly.com/library/view/deep-reinforcement-learning/9781835882702/)
- [Reinforcement Learning](https://learning.oreilly.com/library/view/reinforcement-learning/9781492072386/)
- [Mastering Reinforcement Learning with Python](https://learning.oreilly.com/library/view/mastering-reinforcement-learning/9781838644147/)

### Deep Learning from Scratch 4
- https://github.com/oreilly-japan/deep-learning-from-scratch-4
- [Part 2: Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

```
第 1章　老虎機問題 1
1.1　機器學習的分類與強化學習 1
1.1.1　監督學習 2
1.1.2　無監督學習 2
1.1.3　強化學習 3
1.2　老虎機問題 5
1.2.1　什麼是老虎機問題 5
1.2.2　什麼是好的老虎機 7
1.2.3　使用數學式表示 8
1.3　老虎機演算法 9
1.3.1　價值的估計方法 10
1.3.2　求平均值的實現 12
1.3.3　玩家的策略 15
1.4　老虎機演算法的實現 17
1.4.1　老虎機的實現 17
1.4.2　智慧代理的實現 19
1.4.3　嘗試運行 20
1.4.4　演算法平均的特性 23
1.5　非穩態問題 28
1.5.1　解決非穩態問題前的準備工作 29
1.5.2　解決非穩態問題 32
1.6　小結 34

第 2章　馬可夫決策過程（英語：Markov decision process，MDP）
2.1　什麼是MDP 37
2.1.1　MDP的具體例子 37
2.1.2　智慧代理與環境的互動 39
2.2　環境和智慧代理的數學表示 40
2.2.1　狀態遷移 40
2.2.2　獎勵函數 42
2.2.3　智慧代理的策略 43
2.3　MDP的目標 45
2.3.1　回合制任務和連續性任務 45
2.3.2　收益 46
2.3.3　狀態價值函數 47
2.3.4　 策略和 價值函數 48
2.4　MDP的例子 50
2.4.1　回溯線形圖 51
2.4.2　找出 策略 52
2.5　小結 54

第3章　貝爾曼方程|Bellman Equation
https://velog.io/@jowithu/Bellman-Equation
3.1　貝爾曼方程的推導 57
3.1.1　概率和期望值（推導貝爾曼方程的準備）57
3.1.2　貝爾曼方程的推導 60
3.2　貝爾曼方程的例子 64
3.2.1　有兩個方格的網格世界 64
3.2.2　貝爾曼方程的意義 68
3.3　行動價值函數與貝爾曼方程 68
3.3.1　行動價值函數 69
3.3.2　使用行動價值函數的貝爾曼方程 70
3.4　貝爾曼 方程 71
3.4.1　狀態價值函數的貝爾曼 方程 71
3.4.2　Q函數的貝爾曼 方程 73
3.5　貝爾曼 方程的示例 74
3.5.1　應用貝爾曼 方程 74
3.5.2　得到 策略 76
3.6　小結 78

第4章　動態規劃法 79
4.1　動態規劃法和策略評估 80
4.1.1　動態規劃法簡介 80
4.1.2　嘗試反覆運算策略評估 81
4.1.3　反覆運算策略評估的其他實現方式 86
4.2　解決 大的問題 87
4.2.1　GridWorld類的實現 88
4.2.2　defaultdict的用法 94
4.2.3　反覆運算策略評估的實現 95
4.3　策略反覆運算法 99
4.3.1　策略的改進 99
4.3.2　重複評估和改進 101
4.4　實施策略反覆運算法 102
4.4.1　改進策略 103
4.4.2　重複評估和改進 105
4.5　價值反覆運算法 107
4.5.1　價值反覆運算法的推導 109
4.5.2　價值反覆運算法的實現 113
4.6　小結 116

第5章　蒙特卡洛方法 117
5.1　蒙特卡洛方法的基礎知識 117
5.1.1　骰子的點數和 118
5.1.2　分佈模型和樣本模型 119
5.1.3　蒙特卡洛方法的實現 121
5.2　使用蒙特卡洛方法評估策略 123
5.2.1　使用蒙特卡洛方法計算價值函數 124
5.2.2　求所有狀態的價值函數 126
5.2.3　蒙特卡洛方法的高效實現 129
5.3　蒙特卡洛方法的實現 130
5.3.1　step方法 130
5.3.2　智慧代理類的實現 132
5.3.3　運行蒙特卡洛方法 134
5.4　使用蒙特卡洛方法的策略控制 136
5.4.1　評估和改進 136
5.4.2　使用蒙特卡洛方法實現策略控制 137
5.4.3　ε-greedy演算法（第 1個修改） 139
5.4.4　修改為固定值α的方式（第 2個修改） 141
5.4.5　[ 修改版] 使用蒙特卡洛方法實現策略反覆運算法 142
5.5　異策略型和重要性採樣 145
5.5.1　同策略型和異策略型 145
5.5.2　重要性採樣 146
5.5.3　如何減小方差 150
5.6　小結 152

第6章　TD方法 153
6.1　使用TD方法評估策略 153
6.1.1　TD方法的推導 154
6.1.2　MC方法和TD方法的比較 157
6.1.3　TD方法的實現 158
6.2　SARSA 161
6.2.1　同策略型的SARSA 161
6.2.2　SARSA的實現 162
6.3　異策略型的SARSA 165
6.3.1　異策略型和重要性採樣 166
6.3.2　異策略型的SARSA的實現 167
6.4　Q學習 169
6.4.1　貝爾曼方程與SARSA 170
6.4.2　貝爾曼 方程與Q學習 171
6.4.3　Q學習的實現 173
6.5　分佈模型與樣本模型 175
6.5.1　分佈模型與樣本模型 175
6.5.2　樣本模型版的Q學習 176
6.6　小結 179

第7章　神經網路和Q學習 181
7.1　DeZero簡介 182
7.1.1　使用DeZero 183
7.1.2　多維陣列（張量）和函數 184
7.1.3　 化 186
7.2　線性回歸 189
7.2.1　玩具資料集 189
7.2.2　線性回歸的理論知識 190
7.2.3　線性回歸的實現 191
7.3　神經網路 195
7.3.1　非線性資料集 195
7.3.2　線性變換和啟動函數 196
7.3.3　神經網路的實現 197
7.3.4　層與模型 199
7.3.5　優化器（ 化方法）202
7.4　Q學習與神經網路 204
7.4.1　神經網路的預處理 204
7.4.2　表示Q函數的神經網路 205
7.4.3　神經網路和Q學習 208
7.5　小結 212

第8章　DQN 213
8.1　OpenAI Gym 213
8.1.1　OpenAI Gym的基礎知識 214
8.1.2　隨機智慧代理 216
8.2　DQN的核心技術 218
8.2.1　經驗重播 218
8.2.2　經驗重播的實現 220
8.2.3　目標網路 223
8.2.4　目標網路的實現 224
8.2.5　運行DQN 227
8.3　DQN與Atari 230
8.3.1　Atari的遊戲環境 231
8.3.2　預處理 232
8.3.3　CNN 232
8.3.4　其他技巧 233
8.4　DQN的擴展 234
8.4.1　Double DQN 234
8.4.2　優先順序經驗重播 235
8.4.3　Dueling DQN 236
8.5　小結 238

第9章　策略梯度法 239
9.1　 簡單的策略梯度法 239
9.1.1　策略梯度法的推導 240
9.1.2　策略梯度法的演算法 241
9.1.3　策略梯度法的實現 243
9.2　REINFORCE 248
9.2.1　REINFORCE演算法 249
9.2.2　REINFORCE的實現 250
9.3　基線 251
9.3.1　基線的思路 251
9.3.2　帶基線的策略梯度法 253
9.4　Actor-Critic 254
9.4.1　Actor-Critic的推導 255
9.4.2　Actor-Critic的實現 257
9.5　基於策略的方法的優點 260
9.6　小結 262

第 10章　進一步學習 263
10.1　深度強化學習演算法的分類 263
10.2　策略梯度法的改進演算法 265
10.2.1　A3C和A2C 265
10.2.2　DDPG 268
10.2.3　TRPO和PPO 271
10.3　DQN的改進演算法 272
10.3.1　分類DQN 272
10.3.2　Noisy Network 274
10.3.3　Rainbow 274
10.3.4　在Rainbow以後提出的改進演算法 275
10.4　案例研究 276
10.4.1　棋盤遊戲 277
10.4.2　機器人控制 279
10.4.3　NAS 280
10.4.4　其他案例 282
10.5　深度強化學習的挑戰和可能性 283
10.5.1　應用於實際系統 283
10.5.2　將問題表示為MDP形式時的建議 286
10.5.3　通用人工智慧系統 288
10.6　小結 288
附錄A　異策略型的蒙特卡洛方法 291
附錄B　n-step TD方法 298
附錄C　Double DQN的理解 300
附錄D　策略梯度法的證明 304
後記 308
參考文獻 310

```
