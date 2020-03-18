import numpy as np
import gym
import matplotlib.pyplot as plt
"""we are dealing with continuous state space but since we cant have a
linspace 2 end points and number of bins
digitize: takes buckets and values and gives which bucket the value belongs to"""
position_space = np.linspace(-1.2, 0.6, 20)
"""We want to capture granularity"""
velocity_space = np.linspace(-0.07, 0.07, 20)

#Takes continuous values and digitizes it
def get_state(observation):
 # The observation is position, velocity of a state position action/reset
 #takes an observation and returns a state
    position, velocity = observation #List
    position_bin = int(np.digitize(position, position_space))
    velocity_bin = int(np.digitize(velocity, velocity_space))
    return (position_bin,velocity_bin)


"""
Q-Learning:
Future rewards estimate for a state action pair
1)Initialize state
2) iterate over episode
    a) choose action based on epsilon derived policy
             Draw random number; compare with epsilon; take random action/greedy action
    b) update Q(S,A)=formula ; alpha=> lr ; gamma=> discount factor
             Gamma means how much we value future rewards
"""

# Function defining action in Max_Q policy
def max_action(Q, state, actions=[0,1,2]):
    values= np.array([Q[state,a] for a in actions])
    action=np.argmax(values)
    #Choose action corresponding to max q value
    return action

# This function can be called for any value of a, g, e, and policy(max_q if true else random action if false)
def env_mountain_car( alpha, gamma, epsilon, max_q_policy):
    eps=epsilon
    env=gym.make('MountainCar-v0')
    print( 'Alpha= {} Gamma= {} Epsilon = {}'.format(alpha,gamma, epsilon))
    env._max_episode_steps=5000
    n_games=50000

    states=[]
    # State space in mountain car problem decided by dot product of position and velocity bin
    for position in range(21):
        for velocity in range(21):
            states.append((position, velocity))
    Q = {}
    #Initializing Q table with state and action shape with zeros
    for state in states:
        for action in [0, 1, 2]:
            Q[state, action] = 0


    score=0
    num_of_steps=[] # List of number of steps in all episodes
    total_rewards=np.zeros(n_games)
    Epsilon=np.zeros(n_games)

    for i in range(n_games):
            count=0 #Step size
            done=False # set for each episode
            obs=env.reset() # Assigns random [position, velocity] to obs. => List.
            state=get_state(obs) # Get the specific bins of position and velocity.  => Tuple

            if i%1000==0 and i>0:
                print('episode', i, 'score', score, 'epsilon', eps)
            score=0
            # output only every 1000 episode
            while not done:
                count+=1
                if not max_q_policy:
                    action=np.random.choice([0,1,2])
                else:
                    action = max_action(Q,state)# Take random action from action space
                new_observation, reward, done, info=env.step(action)
                new_state=get_state(new_observation)
                score += reward
                new_action=max_action(Q,new_state)
                Q[state , action]=Q[state , action]+alpha*(reward + gamma*Q[new_state , new_action]-Q[state , action])
                state=new_state
            if done and i%1000==0 and i>0:
                print(' Epsilon at max step of episode {} is {}'.format(i, eps))

            total_rewards[i]=score
            eps=eps-2/n_games if eps>0.01 else 0.01
            Epsilon[i]=eps
            num_of_steps.append(count)

    mean_episode_score_reward=np.zeros(n_games)
    mean_epsilon=np.zeros(n_games)

    for t in range(n_games):
        mean_episode_score_reward[t]= np.mean(total_rewards[max(0,t-50):(t+1)])
        mean_epsilon[t]= np.mean(Epsilon[max(0,t-50):(t+1)])

    print("Average Number of Steps per Episode: " ,(sum(num_of_steps)/n_games) )
    plt.clf()

    plt.plot(mean_episode_score_reward)
    plt.title(' Score by Episodes for intial alpha= {} , gamma= {}, epsilon= {}'.format(alpha,gamma, epsilon))
    plt.xlabel('Episode number')
    plt.ylabel('Score')
    plt.savefig('MountaincarA({}) , gamma= {}, epsilon= {} maxQ {}.png'.format(alpha,gamma, epsilon, max_q_policy))

    plt.clf()

    plt.plot(mean_epsilon)
    plt.title(' Epsilon distribution ')
    plt.xlabel('Episode number')
    plt.ylabel('Epsilon value')
    plt.savefig('Mountaincar Epsilon distribution alpha({}) , gamma= {}, epsilon= {} maxQ {}.png'.format(alpha,gamma, epsilon,max_q_policy))


# Main function that calls open ai gym mountain function
if __name__== '__main__':
    # alphas= [0.1, 0.5, 0.7]
    # gammas=[0.99, 0.5, 0.2]
    # epsilons=[1.0, 0.5, 0.2]
    # for alpha in alphas:
    #     for gamma in gammas:
    #         for epsilon in epsilons:
    #                 env_mountain_car(alpha,gamma, epsilon, max_q_policy=True)
    env_mountain_car(alpha=0.1, gamma=0.99, epsilon=1, max_q_policy=False)

