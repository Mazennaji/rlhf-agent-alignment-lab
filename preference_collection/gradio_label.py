import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import gradio as gr
import imageio
from stable_baselines3 import PPO

from envs.base_env import make_env
from configs.config import CFG
from preference_collection.rollout import generate_pair_with_frames

env = make_env(CFG.env_id, CFG.seed + 300, render_mode="rgb_array")
model = PPO.load("base_agent/checkpoints/base_ppo_final")

state = {"traj_a": None, "traj_b": None, "dataset": []}

os.makedirs("preference_collection/data", exist_ok=True)
os.makedirs("preference_collection/tmp", exist_ok=True)


def frames_to_video(frames, path, fps=30):
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    return path


def new_pair():
    traj_a, traj_b = generate_pair_with_frames(model, env, CFG.trajectory_length)
    state["traj_a"] = traj_a
    state["traj_b"] = traj_b
    video_a = frames_to_video(traj_a["frames"], "preference_collection/tmp/traj_a.mp4")
    video_b = frames_to_video(traj_b["frames"], "preference_collection/tmp/traj_b.mp4")
    return video_a, video_b, f"Labeled pairs so far: {len(state['dataset'])}"


def label_pair(choice):
    if state["traj_a"] is None or state["traj_b"] is None:
        video_a, video_b, status = new_pair()
        return "No active pair, generated a new one.", video_a, video_b, status

    if choice == "A":
        preferred = 1
    elif choice == "B":
        preferred = 0
    else:
        video_a, video_b, status = new_pair()
        return "Skipped.", video_a, video_b, status

    state["dataset"].append({
        "traj_a": {k: v for k, v in state["traj_a"].items() if k != "frames"},
        "traj_b": {k: v for k, v in state["traj_b"].items() if k != "frames"},
        "preferred": preferred,
    })

    with open("preference_collection/data/manual_preference_pairs.pkl", "wb") as f:
        pickle.dump(state["dataset"], f)

    video_a, video_b, status = new_pair()
    return f"Saved. Total labeled: {len(state['dataset'])}", video_a, video_b, status


with gr.Blocks() as demo:
    gr.Markdown("# RLHF Preference Labeling")
    status_box = gr.Textbox(label="Status", interactive=False)
    counter_box = gr.Textbox(label="Progress", interactive=False)

    with gr.Row():
        video_a_box = gr.Video(label="Trajectory A", autoplay=True)
        video_b_box = gr.Video(label="Trajectory B", autoplay=True)

    with gr.Row():
        btn_a = gr.Button("Prefer A")
        btn_b = gr.Button("Prefer B")
        btn_skip = gr.Button("Skip")
        btn_new = gr.Button("New Pair")

    btn_a.click(fn=lambda: label_pair("A"), outputs=[status_box, video_a_box, video_b_box, counter_box])
    btn_b.click(fn=lambda: label_pair("B"), outputs=[status_box, video_a_box, video_b_box, counter_box])
    btn_skip.click(fn=lambda: label_pair("skip"), outputs=[status_box, video_a_box, video_b_box, counter_box])
    btn_new.click(fn=new_pair, outputs=[video_a_box, video_b_box, counter_box])

    demo.load(fn=new_pair, outputs=[video_a_box, video_b_box, counter_box])


if __name__ == "__main__":
    demo.launch()