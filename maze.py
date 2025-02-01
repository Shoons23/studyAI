import numpy as np
import pandas as pd 
import time
import matplotlib.pyplot as plt

class Environment:
    cliff = -3
    road = -1
    goal = 1

    goal_position = [2,2]

    reward_list = [[road, road, road],
                  [road, road, road],
                  [road, road, goal]]
    
    reward_list1 = [["road", "road", "road"],
                   ["road", "road", "road"],
                   ["road", "road", "goal"]]
    
    def __init__(self):
        self.reward = np.asanyarray(self.reward_list)
 
    def move(self, agent, action):
        done = False
        new_pos = agent.pos + agent.action[action]
 
        # 현재 좌표가 목적지점인지 확인
        if self.reward_list[agent.pos[0]][agent.pos[1]] == self.goal:
            reward = self.goal
            observation = agent.set_pos(agent.pos)
            done = True
        
        # 이동 후 좌표가 미로 밖인지 확인
        elif (new_pos[0] < 0) or (new_pos[0] >= self.reward.shape[0]) \
            or (new_pos[1] < 0) or (new_pos[1] >= self.reward.shape[1]):
            reward = self.cliff
            observation = agent.set_pos(agent.pos)
            done = True
 
        # 이동 후 좌표가 미로 내부라면
        else:
            observation = agent.set_pos(new_pos)
            reward = self.reward[observation[0], observation[1]]
 
        return observation, reward, done
 
class Agent:
    action = np.array([[-1,0],[0,1],[1,0],[0,-1]])
    select_action_pr = np.array([0.25,0.25,0.25,0.25])
 
    def __init__(self, initial_pos):
        self.pos = initial_pos
 
    def set_pos(self, position):
        self.pos = position
        return self.pos
    
    def get_pos(self):
        return self.pos
# 상태가치함수
def state_value_function(env, agent, G, max_step, now_step):
    gamma = 0.9
 
    if env.reward_list1[agent.pos[0]][agent.pos[1]] == env.goal:
        return env.goal
 
     # 현재 위치가 목적지?
    if now_step == max_step:
        pos1 = agent.get_pos()
        
        for action in range(len(agent.action)):
            agent.set_pos(pos1)
            observation, reward, done = env.move(agent, action)
            G += agent.select_action_pr[action] * reward
        return G
    
    else:
        pos1 = agent.get_pos()
        
        for action in range(len(agent.action)):
            observation, reward, done = env.move(agent, action)
            G += agent.select_action_pr[action] * reward
 
            if done == True:
                if observation[0] < 0 or observation[0] >= env.reward.shape[0] \
                    or observation[1] < 0 or observation[1] >= env.reward.shape[1]:
                    agent.set_pos(pos1)
 
            next_v = state_value_function(env, agent, 0, max_step, now_step + 1)
            G += agent.select_action_pr[action] * gamma * next_v
            agent.set_pos(pos1)
        
        return G
# 행동가치함수
def action_value_function(env, agent, act, G, max_step, now_step):
    gamma = 0.9

    # 현재위치가 목적지인지 확인하기
    if(env.reward_list1[agent.pos[0]][agent.pos[1]] == "goal"):
        return env.goal
    
    # 마지막 상태에서는 보상값 출력
    if(max_step == now_step):
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act]*reward
        return G

    # 현재 상태의 보상 계산 후 다음 행동과 함께 다음 step 이동
    else:
        pos1 = agent.get_pos()
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act]*reward


        # 이동 후 위치 확인
        if done == True:
            if observation[0] < 0 or observation[0] >= env.reward.shape[0] or \
               observation[1] < 0 or observation[1] >= env.reward.shape[1]:
                agent.set_pos(pos1)

        # 현재 위치 저장
        pos1 = agent.get_pos()

        # 다음 위치로 가능한 모든 행동 선택 후 이동
        for i in range(len(agent.action)):
            agent.set_pos(pos1)
            next_v = action_value_function(env, agent, i, 0, max_step, now_step+1)
            G += agent.select_action_pr[i] * gamma * next_v

        return G 
            
def show_v_table(v_table, env):
    """
    가치 테이블을 터미널에 텍스트로 표시하는 함수
    Args:
        v_table: 상태 가치가 저장된 2D 배열
        env: 환경 정보
    """
    print("\n=== State Value Table ===")
    # 상단 경계선
    print("-" * (v_table.shape[1] * 8 + 1))

    # 각 행을 순회하며 값 출력
    for i in range(v_table.shape[0]):
        print("|", end="")
        for j in range(v_table.shape[1]):
            # 목표 지점인 경우 *로 표시
            if env.reward_list[i][j] == env.goal:
                print(f" *{v_table[i,j]:4.1f} |", end="")
            # cliff인 경우 X로 표시
            elif env.reward_list[i][j] == env.cliff:
                print(f" X{v_table[i,j]:4.1f} |", end="")
            # 일반 지점
            else:
                print(f"  {v_table[i,j]:4.1f} |", end="")
        print()
        # 행 구분선
        print("-" * (v_table.shape[1] * 8 + 1))
    print()

def show_q_table(env, agent, max_step, now_step):
   """
   각 상태에서의 행동가치(Q-value)를 시각화하는 함수
   """
   q_table = np.zeros((env.reward.shape[0], env.reward.shape[1], len(agent.action)))
   
   # 각 상태의 각 행동에 대한 가치 계산
   for i in range(env.reward.shape[0]):
       for j in range(env.reward.shape[1]):
           agent.set_pos([i,j])
           for a in range(len(agent.action)):
               q_table[i,j,a] = action_value_function(env, agent, a, 0, max_step, now_step)

   actions = ['↑', '→', '↓', '←']  
   
   # 그리드 출력
   print("\n=== Combined Grid View ===")
   print("-" * (env.reward.shape[1] * 25 + 1))  # 간격 늘림
   
   for i in range(env.reward.shape[0]):
       # 위쪽 행동가치
       print("|", end="")
       for j in range(env.reward.shape[1]):
           if env.reward_list[i][j] == env.goal:
               print(f"         *{q_table[i,j,0]:6.2f}         |", end="")
           else:
               print(f"          {q_table[i,j,0]:6.2f}         |", end="")
       print()
       
       # 중앙줄 (왼쪽 가치, 최적행동들, 오른쪽 가치)
       print("|", end="")
       for j in range(env.reward.shape[1]):
           max_value = np.max(q_table[i,j])
           best_actions = np.where(np.abs(q_table[i,j] - max_value) < 1e-10)[0]
           action_str = ''.join([actions[a] for a in best_actions])
           
           if env.reward_list[i][j] == env.goal:
               print(f"  {q_table[i,j,3]:6.2f} *{action_str:^6} {q_table[i,j,1]:6.2f}  |", end="")
           else:
               print(f"  {q_table[i,j,3]:6.2f}  {action_str:^6} {q_table[i,j,1]:6.2f}  |", end="")
       print()
       
       # 아래쪽 행동가치
       print("|", end="")
       for j in range(env.reward.shape[1]):
           if env.reward_list[i][j] == env.goal:
               print(f"         *{q_table[i,j,2]:6.2f}         |", end="")
           else:
               print(f"          {q_table[i,j,2]:6.2f}         |", end="")
       print()
       
       print("-" * (env.reward.shape[1] * 25 + 1))  # 간격 늘림

def main():
    env = Environment()
    agent = Agent([0,0])
    max_step_number = 8
    time_len = []

    # 각 상태 가치를 계산
    for max_step in range(max_step_number):
        v_table = np.zeros((env.reward.shape[0], env.reward.shape[1]))
        start_time = time.time()

        for i in range(env.reward.shape[0]):
            for j in range(env.reward.shape[1]):
                agent.set_pos([i,j])
                v_table[i,j] = state_value_function(env, agent, 0, max_step, 0)

        # max_step에 따른 계산 시간 저장
                # max_step에 따른 계산 시간 저장
        time_len.append(time.time()-start_time)
        print(f"max_step_number = {max_step} total_time = {np.round(time.time()-start_time,2)}(s)")
        
        print("\n상태 가치 테이블:")
        show_v_table(np.round(v_table,2), env)
        
        print("\n행동 가치 테이블:")
        show_q_table(env, agent, max_step, 0)


if __name__ == "__main__":
    main()