# Can Induced Emotion Bias LLM Behaviors in Sequential Decision Making?

**Minh Khoi Ho<sup>1</sup>, Zihao Zhu<sup>2</sup>, Runchuan Zhu<sup>3</sup>, Levina Li<sup>4</sup>, Zhiwen Fan<sup>2</sup>, Zhangyang Wang<sup>5</sup>, Junyuan Hong<sup>6</sup>**

<sup>1</sup>MBZUAI&emsp;<sup>2</sup>Texas A&M University&emsp;<sup>3</sup>National University of Singapore&emsp;<sup>4</sup>UCLA&emsp;<sup>5</sup>University of Texas at Austin&emsp;<sup>6</sup>Massachusetts General Hospital, Harvard University

# Abstract

![Overview](EmotionAgent.png)

As Large Language Models (LLMs) are increasingly deployed as autonomous agents in high-stakes domains, understanding contextual factors that may modulate their decision-making becomes critical. While LLMs are trained to perceive and resonate with users' emotions, it remains unclear whether induced emotion can influence their sequential decision-making. We investigate this question using the Iowa Gambling Task (IGT), a classic psychological paradigm for studying decision-making under uncertainty, combined with an imagination-based emotion induction procedure. We first validate the feasibility of this paradigm by confirming that LLMs can sense strong, distinguishable emotions from context and that LLM agents can learn from sequential interactions in a human-like pace. With the validated setup, we find that, different from humans, induced emotion does not significantly bias the decision dynamics of LLM agents on average. However, the effects of anger are conditioned: inducing anger makes LLM agents less sensitive to penalties for bad decisions, and in early stages of the game, anger can lower exploration, locking decisions into a few choices early. These findings reveal the subtle yet distinct effects of induced emotion on LLM decision-making compared to human behavior, and provide a tool for future research on affective modulation of LLM agents.

