# RL_TermProject-DDPG
This is a project to learn MLAgent by applying the DDPG algorithm.   
The base code is forked from [here](https://github.com/spiz26/RL_TermProject) provied in Seoultech RL lecture in Late 2023.   

## Code
### DDPG class
```
class DDPG_1:
    def __init__(self, state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        ...
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
```
most important variables are *actor*, *critic*, *noise*
### selecting action at each step
```
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze().numpy()
        noise = self.noise.sample()
        result = np.clip(action + noise, -1, 1) # assuming action space is -1 to 1
        return result
```
Compute action from state tensor. The noise is OrnsteinUhlenbeckNoise.
### update Q-value
```
    def update(self, state, action, reward, next_state, done):
        ...
        # Compute the Q-value
        target_value = reward + (1.0 - done) * self.gamma * self.critic(next_state, self.actor(next_state))

        # Update the critic
        ...
        # Update the actor
        ...
```
If the episode is over the Q-value is just reward. During the episode(agent is driving without collision) the q-value would calculated as below.   
$$L(\phi,D)=\underset{(s,a,r,s',d)~\mathcal{D}}{E}[(Q_\phi(s,a)-(r+\gamma(1-d)\underset{a'}{\mathrm{max}Q^*(s',a'}))]$$

## result
The agent is not learning well: after about 100,000 epochs of training, the agent is still not moving forward turning right and crash with the wall. This indicates a problem in the code, and I'm planning to fix it. I need to learn more about the algorithm.

# RL_TermProject
|   <span style = "background-color : lightgray">MDP</span>  |                                                                                          <center>Contents</center>                                                                                         |
|:------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| State  | ① Velocity of Kart <br> ② Rays of distance between kart and the object * 9 <br> ③ Inner product value between the next destination and the kart direction vector of the kart <br> ④ Kart acceleration or not |
| Action | ① [left, straight, right] <br> ② [accleration, stop, brake]                                                                                                                                    |
| Reward | ① hit: -2 <br> ② pass next goal: 1 <br> ③ velocity of kart is 0: -0.002 <br> ④ Inner product value: 0.03                                                                                                 |

state는 위 표의 state를 4개 쌓은 벡터를 사용한다.
```python
current_state = decision_steps.obs[0]
next_state = terminal_steps.obs[0]
```
