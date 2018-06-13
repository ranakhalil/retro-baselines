### How I soared high training with sonic, then crash landed in a Deep Q-world

#### The summary:

When I started training Sonic via Rainbow DQN, jerk agent and ppo, I noticed the more you train sonic it doesn't necessairly mean sonic gets better and smarter. 

During each training set of steps, sonic seems to perform really well, then get exponentially worse and end there. 

Seeing those results I was left with few interesting choices:
1- Reduce the number of steps to train Sonic
2- Reduce the number of epochs to train Sonic for n-steps
3- Utilize and test a variety of learning rates and epsilons to tweak the rate at which Sonic will learn.
4- Build something simple that incrementally you can train sonic and understand why that behavior happens in repition
5- Build tools that help Sonic pause or freeze model, once it reaches an optimal level of training

#### What actually happened:

In the begining: I started the competition looking through a simple agent implementation, the first question I was trying to answer was: What is the env I am using to train my agent, and how can I test?

Through the very simple example here:

```
from retro_contest.local import make


def main():
    env = make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    obs = env.reset()
    while True:
        # obs, rew, done, info = env.step(env.action_space.sample())
        # env.render()
        # if done:
        #     obs = env.reset()
        action = env.action_space.sample()
        #  ["X", "Z", "TAB", "ENTER", "UP", "DOWN", "LEFT", "RIGHT", "C", "A", "S", "D"],
        action[7] = 1
        action[4] = 1 #UP
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            print('episode complete')
            env.reset()


if __name__ == '__main__':
    main()
```

I was able to see Sonic navigating, and jumping and fulfilling certain steps I was feeding.

The next step was to experiment with a variety of baselines scripts. I started with the JERK script:

```
EMA_RATE = 0.1
EXPLOIT_BIAS = 0.15
TOTAL_TIMESTEPS = int(1e6)

def main():
    """Run JERK on the attached environment."""
    
    # env = make(
    #     game='SonicTheHedgehog-Genesis', 
    #     state='SpringYardZone.Act2', 
    #     scenario='contest'
    # )

    env = grc.RemoteEnv('tmp/sock')
    env = TrackedEnv(env)
    new_ep = True
    solutions = []
    while True:
        if new_ep:
            if (solutions and
                    random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, best_pair[1])
                best_pair[0].append(new_rew)
                print('replayed best with reward %f' % new_rew)
                continue
            else:
                env.reset()
                new_ep = False
        rew, new_ep = move(env, 100)
        if not new_ep and rew <= 0:
            print('backtracking due to negative reward: %f' % rew)
            _, new_ep = move(env, 70, left=True)
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))

def move(env, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
    while not done and steps_taken < num_steps:
        action = np.zeros((12,), dtype=np.bool)
        action[6] = left
        action[7] = not left
        if jumping_steps_left > 0:
            action[0] = True
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action[0] = True
        _, rew, done, _ = env.step(action)
        # env.render()
        total_rew += rew
        steps_taken += 1
        if done:
            break
    return total_rew, done
```

What was really awesome about the jerk agent as its name indicates, how you can learn enough and still make decent progress navigating through the levels.

Now that I have gained a lot more confidence, and reading through the tech report, realized to level up in the competition experimenting with PPO and Rainbow DQN is a must.

Experimenting with both Rainbow DQN and PPO, noticed I was getting pretty good results with Rainbow DQN vs PPO. In which case, I decided to deciate the rest of the month and a half at this point to tune and look exclusively into Rainbow DQN.

### Rainbow DQN Baseline:

Through repeated training with Rainbow DQN, I noticed pretty fast, at a certain level of training Sonic seems to suffer a cognition overload and does worse in training to go through the levels.

Which brought me to the next step to introspect the values I have used tunning the rainbow DQN baseline:

```
def main():
    """Run DQN until the environment throws an exception."""
    env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 5)
        optimize = dqn.optimize(learning_rate=1e-5, epsilon=1.5e-4)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=2100000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.5),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=5192,
                  batch_size=32,
                  min_buffer_size=20000)
```

The few parameters that really made a huge difference tunning where firstly the `min_val` , `max_val`, `learning_rate` , `epsilon` and `num_steps`. The more I have increased the number of steps, the more promising I found some of the results. However, HOWEVER , it took an exceptionally longer amount of time to test and verify the results of my tweaked Rainbow DQN.

Until one fateful moment, where my blue friend Sonic did so well achieving an score of `4354.68` which placed Sonic at a higher place on the leaderboard around the first twenty top spots.

Here is a video to demonstrate.
[![Highest Score](http://img.youtube.com/vi/zkqBV34kCOM/0.jpg)](https://www.youtube.com/embed/zkqBV34kCOM "SonicHedgehog higest score")

Feeling confident and strong , I decided to tweak my learning steps, learning rate and min and max values. At the same time, I went ahead and re-kicked the same job that generated the highest score to see how I can repeat the same score if not higher. I however noticed pretty fast, the more I have trained the same docker image (with the higest score) the less I was able to reproduce the high score again with the same code. Sometimes I have even gotten a score that is so much lower than the scores I have expected before.

While I wasn't sure why running the same code on the same hardware have yielded different results. I went ahead training locally and re-tweaking the parameters. At the end of the competition I was able to get a score within the 3k range, however , not within the 4k range rewards I have anticipated to go back to.


#### Lessons Learnt:

One of the main lessons I have learnt throughout the competition which made a huge difference in scaling and speeding training sonic, was how to customize the reward function.

Here is the example of the custom reward function I have used with Rainbow DQN:

```

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        if reward > 2500:
            return reward * 0.09
        elif reward > 1700:
            return reward *  0.07
        else:
            return reward * 0.05
```

One of the main lessons I have learnt, thinking about how the game is structures is giving higher scales of reward when sonic hits higher reward items. It helped me scale my learning or rewards linearly, I can see sonic improving in the begining and learning faster.


#### So what now?

Well, I will ceratinly read all the awesome blog posts my peers have written for their implementation, and continue learning and experimenting with the retro gym :) 
To OpenAI: thank you so much for the opportunity!
