# Install once
install.packages("DiagrammeR")

# Load package
library(DiagrammeR)

grViz("

digraph MDP_Methodology {

graph[
layout = dot,
rankdir = TB,
bgcolor = white,
labelloc = t,
fontsize = 26,
fontname = Helvetica,
label = 'MARKOV DECISION PROCESS (MDP) METHODOLOGY'
]

node[
shape = rectangle,
style = 'rounded,filled',
fontname = Helvetica,
fontsize = 15,
width = 4.8,
height = 1.0,
color = '#1F4E79',
penwidth = 2
]

edge[
color = '#2C3E50',
penwidth = 2.5,
arrowsize = 0.8
]

A[
label='Input Dataset

• Collect patient records
• Clinical symptoms
• Medical history
• Laboratory reports',
fillcolor='#AED6F1'
]

B[
label='State Representation

• Define health states
• Encode patient features
• Create state space
• Normalize attributes',
fillcolor='#ABEBC6'
]

C[
label='Action Selection

• Select treatment action
• Drug recommendation
• Diagnostic decision
• Policy exploration',
fillcolor='#FCF3CF'
]

D[
label='State Transition Model

• Predict next state
• Transition probabilities
• Environment dynamics
• Update patient condition',
fillcolor='#FAD7A0'
]

E[
label='Reward Function

• Positive reward for recovery
• Negative reward for risk
• Maximize long-term outcome
• Minimize treatment cost',
fillcolor='#F5B7B1'
]

F[
label='Policy Evaluation

• Estimate value function
• Compare possible actions
• Improve decision policy
• Compute expected reward',
fillcolor='#D7BDE2'
]

G[
label='Optimal Policy Generation

• Select best action
• Maximize cumulative reward
• Optimize treatment strategy
• Generate final policy',
fillcolor='#F8C471'
]

H[
label='Performance Evaluation

• Accuracy
• Precision
• Recall
• F1-Score
• Decision effectiveness',
fillcolor='#82E0AA'
]

A -> B
B -> C
C -> D
D -> E
E -> F
F -> G
G -> H

subgraph cluster1 {

label = 'MDP Decision-Making Architecture'

fontsize = 20
fontcolor = white
color = '#154360'
fillcolor = '#EBF5FB'
style = 'rounded,filled'
penwidth = 4

A;B;C;D;E;F;G;H;

}

}
")
