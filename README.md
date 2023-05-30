# TransformerLight —— Official Implementation
## TransformerLight: A Novel Sequence Modeling Based Traffic Signaling Mechanism via Gated Transformer

**Abstract:** Traffic signal control (TSC) is still one of the most significant and challenging research problems in the transportation field.
Reinforcement learning (RL) has achieved great success in TSC but suffers from critically high learning costs in practical applications due to the excessive trial-and-error learning process.
Offline RL is a promising method to reduce learning costs whereas the data distribution shift issue is still up in the air.
To this end, in this paper, we formulate TSC as a sequence modeling problem with a sequence of Markov decision process described by states, actions, and rewards from the traffic environment.
A novel framework, namely TransformerLight, is introduced, which does not aim to fit into value functions by averaging all possible returns, but produces the best possible actions using a gated Transformer.
Additionally, the learning process of TransformerLight is much more stable by replacing the residual connections with gated transformer blocks due to a dynamic system perspective. Through numerical experiments on offline datasets, we demonstrate that the TransformerLight model: (1) 
can build a high-performance adaptive TSC model without dynamic programming; (2) achieves a new state-of-the-art compared to most published offline RL methods so far; and (3) shows a more stable learning process than offline RL and recent Transformer-based methods. The relevant dataset and code are available at https://github.com/Smart-Trafficlab/TransformerLight.

## Experiment


**run_DT.py, run_TT.py, run_decision_transforlight.py:**
Scripts that run traffic simulation experiments using specified traffic scenarios. 


**run_test.py:**
Implementing a traffic simulation testing framework for evaluating trained traffic models using different traffic scenarios.

**summary.py:** 
Analyzing and summarizing test results of RL algorithms and conventional methods, calculating indicators such as duration and number of vehicles, and generating summary reports.