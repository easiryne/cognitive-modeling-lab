import random

import gradio as gr
import numpy as np

# Bandit task configuration
bandit_trajectory = np.load('bandit_data.npy')
n_trials = bandit_trajectory.shape[-1]
actions = []
rewards = []
score = 0
trial = 0

# Function to simulate pulling an arm
def pull_arm(arm):
    global trial, actions, score
    if trial < n_trials:
        reward = bandit_trajectory[arm, trial]
        actions.append(arm)
        rewards.append(reward)
        trial += 1
        score += reward
        return f"Trial {trial}/{n_trials}  -  Reward: {reward:.2f}\n{'-'*10}\nScore: {score:.2f}"
    else:
        msg = save_rewards()
        return f"All trials completed! \nYour score is: {score:.2f}\n{msg}"

def save_rewards():
    np.save('my_actions.npy', np.array(actions))
    np.save('my_rewards.npy', np.array(rewards))
    return "Rewards saved under  'my_actions.npy'  and  'my_rewards.npy'"

def reset():
    global trial, actions, rewards, score
    trial = 0
    actions = []
    rewards = []
    score = 0

# Gradio interface with clickable buttons for both arms
with gr.Blocks(css=".red-button { color: red; }") as interface:
    gr.Markdown("# Two-Armed Restless Bandit Task")
    gr.Markdown("Click on a button to select an arm.\nTry to maximize your reward!")

    with gr.Row():  # Create a row with two buttons side by side
        arm_1_btn = gr.Button("Left")
        arm_2_btn = gr.Button("Right")

    result_output = gr.Textbox(label="Result", interactive=False)

    # Set up button click events (no submit required)
    arm_1_btn.click(fn=lambda: pull_arm(0), inputs=[], outputs=result_output)
    arm_2_btn.click(fn=lambda: pull_arm(1), inputs=[], outputs=result_output)

    gr.Button("Reset", elem_classes="red-button").click(fn=reset, inputs=[], outputs=result_output)

# Launch the interface on localhost
interface.launch(share=False)