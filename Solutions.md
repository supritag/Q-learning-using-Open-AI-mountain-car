## Understandings: 
#### 1) Compared to baseline (score=-200) RL gave a score around -129 
#### 2) states are position and velocity discretized bins and the dot product of the 2 spaces
   actions are 0, 1, 2 where 0: left 1: neutral 2: right movements
   size of Q table is 400 3
#### 3)  for Q learning arbitrarily we take :
               alpha learning rate around 0.1 to track small updates in q table ;
               gamma=> discount factor
     Gamma means how much we value future rewards a value around 1 means we rely a lot on future rewards to update our policy 
#### 4) We are also using random instead of q table max val action
#### 5) epsilon decay is chosen according to 3/number of episodes at constant rate
#### 6) average number of steps per episode
             eg. when Alpha= 0.1 Gamma= 0.99 Epsilon = 1.0
                  Average Number of Steps per Episode:  570.92346
#### 7) Q learning uses value based iteration as it is off policy algorithm
#### 8) Keeping the constraints of the problem expected lifetime value depends on current state and historic actions and historic rewards

## Abstract:
We see that the mountain car problem is a classical open AI gym problem to optimize the score in the given maximum steps
and reach the flag to solve for goal. We do this by using off policy Q learning that updates Q table according to action driven by epsilon and max Q policy.
We take into account the learning rate, discount factor and achieve better outcome
from -200 to -129 score enhancing baseline with the tuning.

## Inference
The different values of episilon, score were observed at different combinations of hyperparameters.
While the entire values have been summarized in values. txt and different graphs for score and epsilon traversals across episodes. 
It is interesting to look at some key values  below:

         When
         Alpha= 0.1 Gamma= 0.99 Epsilon = 1.0
         lowest score is score -129.0 highest -237
         Average Number of Steps per Episode:  143.72618
         we arrive at e=0.01 at episode 25000

         When:

         Alpha= 0.1 Gamma= 0.99 Epsilon = 0.5
         lowest score is score -131.0 highest -237
         Average Number of Steps per Episode:  144.02276
         we arrive at e=0.01 at episode 13000

         When:
         Alpha= 0.1 Gamma= 0.99 Epsilon = 0.2
         lowest score is score -128.0 highest -258
         Average Number of Steps per Episode:  139.37876
         we arrive at e=0.01 at episode 5000



## References:
Ratio of self contribution vs outside help around 40:60

   1) https://www.youtube.com/watch?v=Bi-CKm9zS9c&t=331s
   
   2) https://www.youtube.com/watch?v=3zeg7H6cAJw
   
   3) https://github.com/philtabor/Reinforcement-Learning-In-Motion/blob/master/Unit-8-The-Mountaincar/mountainCar.py
   
   4) https://gym.openai.com/envs/MountainCar-v0/
   
   5) https://github.com/openai/gym/wiki/MountainCarContinuous-v0
   
   6) https://towardsdatascience.com/reinforcement-learning-tutorial-with-open-ai-gym-9b11f4e3c204
   


## MIT License

Copyright (c) 2020 Suprita Ganesh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


