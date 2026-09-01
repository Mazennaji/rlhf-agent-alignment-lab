import numpy as np


def collect_trajectory(model, env, max_steps, deterministic=False):
    obs, info = env.reset()
    observations, actions, rewards = [], [], []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs
        if terminated or truncated:
            break

    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "total_return": float(np.sum(rewards)),
    }


def safe_render(env, last_good_frame):
    try:
        frame = env.render()
        if frame is not None:
            return frame
    except Exception:
        pass
    return last_good_frame


def collect_trajectory_with_frames(model, env, max_steps, deterministic=False):
    obs, info = env.reset()
    observations, actions, rewards, frames = [], [], [], []

    first_frame = safe_render(env, None)
    if first_frame is None:
        first_frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frames.append(first_frame)

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        frames.append(safe_render(env, frames[-1]))

        obs = next_obs
        if terminated or truncated:
            break

    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "total_return": float(np.sum(rewards)),
        "frames": frames,
    }


def generate_pair(model, env, max_steps):
    traj_a = collect_trajectory(model, env, max_steps, deterministic=False)
    traj_b = collect_trajectory(model, env, max_steps, deterministic=False)
    return traj_a, traj_b


def generate_pair_with_frames(model, env, max_steps):
    traj_a = collect_trajectory_with_frames(model, env, max_steps, deterministic=False)
    traj_b = collect_trajectory_with_frames(model, env, max_steps, deterministic=False)
    return traj_a, traj_b