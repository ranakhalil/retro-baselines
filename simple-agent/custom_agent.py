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